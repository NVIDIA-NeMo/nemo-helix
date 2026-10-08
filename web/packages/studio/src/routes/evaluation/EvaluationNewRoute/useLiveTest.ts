// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { resolveKeyPath } from '@nemo/common/src/utils/file';
import { evalsRunEvaluateLiveEvaluation } from '@nemo/sdk/generated/evals/evals-plugin-live-evaluation-route';
import type { LiveScoreRequest, MetricInline } from '@nemo/sdk/generated/evals/schema';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { buildMetricBundles } from '@studio/routes/evaluation/EvaluationNewRoute/buildEvaluationSpec';
import {
  composeGenerationPrompt,
  type EvaluationFormValues,
  type DatasetBindings,
  toFieldMapping,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { getModelInferenceGatewayUrl } from '@studio/util/models';
import { useCallback, useRef, useState } from 'react';

export interface LiveTestResult {
  /** What the model actually answered, so a bad score can be read against it. */
  output: string;
  scores: { name: string; value: number; label?: string }[];
  /** Metrics that failed while their siblings scored. */
  errors: { metric: string; message: string }[];
}

export type LiveTestState =
  | { status: 'idle' }
  | { status: 'busy'; label: string }
  | { status: 'done'; result: LiveTestResult }
  | { status: 'error'; message: string };

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : undefined;

const statusOf = (error: unknown): number | undefined => {
  const status = asRecord(error)?.status;
  if (typeof status === 'number') return status;
  const responseStatus = asRecord(asRecord(error)?.response)?.status;
  return typeof responseStatus === 'number' ? responseStatus : undefined;
};

/** The server's own explanation, which already names the culprit: "target
 *  generation failed: ..." for the model under test, "llm-judge: ..." per failed
 *  metric. A 422 carries FastAPI's list of ``{msg}`` instead of a string. */
const detailOf = (error: unknown): string | undefined => {
  const record = asRecord(error);
  const body = asRecord(record?.error) ?? asRecord(asRecord(record?.response)?.data);
  const raw = body?.detail ?? body?.message ?? record?.message;
  const candidate = Array.isArray(raw)
    ? raw
        .map((entry) => asRecord(entry)?.msg)
        .filter((msg): msg is string => typeof msg === 'string')
        .join('; ')
    : raw;
  if (typeof candidate !== 'string' || !candidate.trim()) return undefined;
  return candidate.length > 240 ? `${candidate.slice(0, 240)}…` : candidate;
};

const describeFailure = (error: unknown): string => {
  const status = statusOf(error);
  const detail = detailOf(error);
  if (status) return `The live test failed (${status}).${detail ? ` ${detail}` : ''}`;
  return `Could not reach the evaluator.${detail ? ` ${detail}` : ''}`;
};

const bareModelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

/** Over a single row the aggregate mean IS that row's score, so it lands exactly
 *  on a rubric value or on NaN. */
const rubricLabel = (
  values: EvaluationFormValues,
  name: string,
  value: number
): string | undefined => {
  const scoreName = name.split('.').pop();
  const score = values.body.scores.find((entry) => entry.name === scoreName);
  if (score?.scoreType !== 'rubric') return undefined;
  return score.rubric.find((item) => item.value === value)?.label;
};

/**
 * Scores one row through the evaluator's ``evaluate/live`` route, which
 * generates the response and scores it in-process: no job, nothing persisted.
 *
 * The request is built the way a real run's is -- same target, prompt template,
 * field mapping and metrics -- so the live test exercises what the user
 * configured rather than a bare chat call.
 */
export function useLiveTest() {
  const workspace = useWorkspaceFromPath();
  const [state, setState] = useState<LiveTestState>({ status: 'idle' });
  /** Guards against a superseded run overwriting a newer one's state. */
  const runRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    runRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ status: 'idle' });
  }, []);

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

      const input = bindings.inputPath ? resolveKeyPath(row, bindings.inputPath) : null;
      if (typeof input !== 'string' || !input) {
        setState({ status: 'error', message: 'The mapped Input field is empty for this row.' });
        return;
      }

      setState({ status: 'busy', label: 'Generating and scoring a response…' });

      const fieldMapping = toFieldMapping(values.fieldMapping);
      const request: LiveScoreRequest = {
        dataset: [row],
        metrics: buildMetricBundles(values, bindings, workspace) as unknown as MetricInline[],
        target: {
          url: getModelInferenceGatewayUrl(workspace, values.model),
          name: bareModelName(values.model),
        },
        prompt_template: composeGenerationPrompt(bindings),
        ...(fieldMapping ? { field_mapping: fieldMapping } : {}),
      };

      try {
        const response = await evalsRunEvaluateLiveEvaluation(
          workspace,
          request,
          controller.signal
        );
        if (superseded()) return;
        setState({
          status: 'done',
          result: {
            output: response.output ?? '',
            scores: response.metrics.flatMap((metric) =>
              (metric.scores ?? []).map((score) => {
                const value = score.mean ?? Number.NaN;
                return { name: score.name, value, label: rubricLabel(values, score.name, value) };
              })
            ),
            errors: response.metrics.flatMap((metric) =>
              metric.error ? [{ metric: metric.metric, message: metric.error }] : []
            ),
          },
        });
      } catch (error) {
        if (superseded()) return;
        setState({ status: 'error', message: describeFailure(error) });
      }
    },
    [workspace]
  );

  return { state, run, cancel };
}
