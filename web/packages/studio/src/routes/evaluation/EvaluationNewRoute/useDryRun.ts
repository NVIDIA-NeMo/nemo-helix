// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { resolveKeyPath } from '@nemo/common/src/utils/file';
import { logger } from '@nemo/common/src/utils/logger';
import {
  evaluatorCreateEvaluateJob,
  evaluatorDeleteEvaluateJob,
  evaluatorGetEvaluateJob,
  evaluatorListEvaluateJobResults,
} from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import {
  type EvaluateJobRequest,
  type MetricInline,
  PlatformJobStatus,
} from '@nemo/sdk/generated/evaluator/schema';
import { buildEvalJobName } from '@studio/components/evaluation/submitEvaluationJob';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { buildMetricBundles } from '@studio/routes/evaluation/EvaluationNewRoute/buildEvaluationSpec';
import {
  type EvaluationFormValues,
  type DatasetBindings,
  toFieldMapping,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useCallback, useRef, useState } from 'react';
import { useAuth } from 'react-oidc-context';

const TERMINAL_STATUSES: string[] = [
  PlatformJobStatus.completed,
  PlatformJobStatus.error,
  PlatformJobStatus.cancelled,
];

const POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 120_000;

export interface DryRunResult {
  /** What the model actually answered, so a bad score can be read against it. */
  output: string;
  scores: { name: string; value: number }[];
}

export type DryRunState =
  | { status: 'idle' }
  | { status: 'busy'; label: string }
  | { status: 'done'; result: DryRunResult }
  | { status: 'error'; message: string };

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;

const statusOf = (error: unknown): number | undefined => {
  const status = asRecord(error)?.status;
  if (typeof status === 'number') return status;
  const responseStatus = asRecord(asRecord(error)?.response)?.status;
  return typeof responseStatus === 'number' ? responseStatus : undefined;
};

/** The server's own explanation, which carries what a bare status code cannot --
 *  a gateway 502 wraps the upstream reason, e.g. "Backend returned 404: Function
 *  ... Not found for account". */
const detailOf = (error: unknown): string | undefined => {
  const record = asRecord(error);
  const body = asRecord(record?.error) ?? asRecord(asRecord(record?.response)?.data);
  const candidate = body?.detail ?? body?.message ?? record?.message;
  if (typeof candidate !== 'string' || !candidate.trim()) return undefined;
  return candidate.length > 240 ? `${candidate.slice(0, 240)}…` : candidate;
};

/** Names the model that failed. The dry run calls two different models, and
 *  "returned 503" is not actionable without knowing which one to swap. */
const describeModelFailure = (role: string, modelRef: string, error: unknown): string => {
  const status = statusOf(error);
  const raw = detailOf(error);
  const detail = status && raw?.startsWith(`${status} `) ? raw.slice(`${status} `.length) : raw;
  const subject = `${role} (${bareModelName(modelRef)})`;
  if (status) return `${subject} returned ${status}.${detail ? ` ${detail}` : ''}`;
  return `Could not reach ${subject}.${detail ? ` ${detail}` : ''}`;
};

const bareModelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

/** The aggregate-scores artifact is a LIST of per-score statistics, not a nested
 *  record. Verified against a live run: {"scores":[{"name":"exact-match.exact-match",
 *  "mean":1.0,"count":1,...}]}. Over a single dry-run row the mean IS the row's
 *  score. (Note useEvaluationJobResultV2 types this as a nested record, which does
 *  not match what the API returns.) */
const flattenScores = (payload: {
  scores?: { name?: string; mean?: number | null }[];
}): { name: string; value: number }[] =>
  (payload.scores ?? [])
    .filter((score) => typeof score.name === 'string')
    .map((score) => ({ name: score.name as string, value: score.mean ?? Number.NaN }));

/**
 * Scores one row, in two phases.
 *
 * Inference is issued by Studio rather than by the job, because a 502 from the
 * model endpoint is the common failure today and inside a job it costs three
 * retries with backoff before surfacing as an opaque job error -- and with
 * `ignore_request_failure` on it would surface as a NaN score and a SUCCEEDING
 * job. Calling directly fails immediately, names the status code, and creates no
 * job at all on the failure path.
 *
 * Scoring then runs OFFLINE against the generated answer: `build_offline_sample`
 * reads `row.output` and synthesises `sample.output_text`, so the metric
 * templates are identical to a real online run.
 *
 * The job is deleted once read. A dry run is an affordance, not a run of record.
 */
export function useDryRun() {
  const workspace = useWorkspaceFromPath();
  const auth = useAuth();
  const [state, setState] = useState<DryRunState>({ status: 'idle' });
  /** Guards against a superseded run overwriting a newer one's state. */
  const runRef = useRef(0);
  const jobRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const discardJob = useCallback(
    (name: string) => {
      // Fire and forget: the score is already read, and a failed cleanup must not
      // present as a failed dry run.
      void evaluatorDeleteEvaluateJob(workspace, name).catch((error) => {
        logger.error(`Dry run: could not delete ephemeral job ${name}: ${String(error)}`);
      });
    },
    [workspace]
  );

  /** Bumping the run id makes every later `superseded()` check bail, which stops
   *  the poll loop without unwinding it. The job is discarded here rather than
   *  left to the loop, so cancelling cannot leak the very thing the dry run
   *  promises to clean up. */
  const cancel = useCallback(() => {
    runRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    if (jobRef.current) {
      discardJob(jobRef.current);
      jobRef.current = null;
    }
    setState({ status: 'idle' });
  }, [discardJob]);

  const run = useCallback(
    async (
      values: EvaluationFormValues,
      bindings: DatasetBindings,
      row: Record<string, unknown>
    ) => {
      const runId = ++runRef.current;
      const superseded = () => runRef.current !== runId;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      // A previous run's job is now irrelevant; do not leave it behind.
      if (jobRef.current) {
        discardJob(jobRef.current);
        jobRef.current = null;
      }

      const input = bindings.inputPath ? resolveKeyPath(row, bindings.inputPath) : null;
      if (typeof input !== 'string' || !input) {
        setState({ status: 'error', message: 'The mapped Input field is empty for this row.' });
        return;
      }

      setState({ status: 'busy', label: 'Generating a response…' });

      let output: string;
      try {
        const completion = await createChatCompletion({
          model: bareModelName(values.model),
          workspace,
          messages: [{ role: 'user', content: input }],
          accessToken: auth.user?.access_token,
          signal: controller.signal,
        });
        output = ('choices' in completion ? completion.choices[0]?.message?.content : null) ?? '';
        if (!output) {
          setState({
            status: 'error',
            message: `The model being evaluated (${bareModelName(values.model)}) returned an empty response.`,
          });
          return;
        }
      } catch (error) {
        if (superseded()) return;
        setState({
          status: 'error',
          message: describeModelFailure('The model being evaluated', values.model, error),
        });
        return;
      }

      if (superseded()) return;

      // Pre-flight the judge for the same reason we generate here rather than in
      // the job: inside the job a judge 502/429 costs the full retry budget and
      // then surfaces as "Job exited with code 1", with the real status buried in
      // stderr. One trivial call names it immediately and creates no job.
      // Verified live: unavailable models return 502 (upstream 404) and an
      // exhausted account returns 429.
      if (values.body.metrics['llm-judge']) {
        setState({ status: 'busy', label: 'Checking the judge model…' });
        try {
          await createChatCompletion({
            model: bareModelName(values.body.judgeModel),
            workspace,
            messages: [{ role: 'user', content: 'ping' }],
            max_tokens: 1,
            accessToken: auth.user?.access_token,
            signal: controller.signal,
          });
        } catch (error) {
          if (superseded()) return;
          setState({
            status: 'error',
            message: describeModelFailure('The judge model', values.body.judgeModel, error),
          });
          return;
        }
      }

      if (superseded()) return;
      setState({ status: 'busy', label: 'Scoring the row…' });

      const name = buildEvalJobName('livetest');
      const request: EvaluateJobRequest = {
        name,
        spec: {
          // Inline single row, offline: no target, so nothing re-runs inference.
          dataset: [{ ...row, output }],
          metrics: buildMetricBundles(values, bindings, workspace) as unknown as MetricInline[],
          field_mapping: { ...(toFieldMapping(values.fieldMapping) ?? {}), output: 'output' },
        },
      };

      try {
        const job = await evaluatorCreateEvaluateJob(workspace, request);
        jobRef.current = job.name;

        const deadline = Date.now() + POLL_TIMEOUT_MS;
        let status: string | undefined;
        while (Date.now() < deadline) {
          if (superseded()) return;
          await sleep(POLL_INTERVAL_MS);
          const polled = await evaluatorGetEvaluateJob(workspace, job.name);
          status = polled.status ?? undefined;
          if (status && TERMINAL_STATUSES.includes(status)) break;
        }

        if (superseded()) return;

        if (status !== PlatformJobStatus.completed) {
          setState({
            status: 'error',
            message:
              status === PlatformJobStatus.error
                ? 'Scoring failed. Both models responded, so check the metric configuration.'
                : 'Scoring did not finish in time.',
          });
          discardJob(job.name);
          jobRef.current = null;
          return;
        }

        const results = await evaluatorListEvaluateJobResults(workspace, job.name);
        const aggregate = results?.data?.find((entry) => entry.name === 'aggregate-scores');
        const scores = aggregate?.download_url
          ? flattenScores(await (await fetch(aggregate.download_url)).json())
          : [];

        if (superseded()) return;
        // Read before delete: once the score is in hand the job has no further use.
        setState({ status: 'done', result: { output, scores } });
        discardJob(job.name);
        jobRef.current = null;
      } catch (error) {
        if (superseded()) return;
        setState({
          status: 'error',
          message: `Could not score the row: ${String((error as Error)?.message ?? error)}`,
        });
        if (jobRef.current) {
          discardJob(jobRef.current);
          jobRef.current = null;
        }
      }
    },
    [workspace, auth.user?.access_token, discardJob]
  );

  return { state, run, cancel };
}
