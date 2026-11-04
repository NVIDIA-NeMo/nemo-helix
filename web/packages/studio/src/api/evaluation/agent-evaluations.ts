// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import {
  evalsCreateAgentEvaluateJob,
  evalsGetAgentEvaluateJob,
  evalsListAgentEvaluateJobs,
} from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-jobs-routes';
import {
  evalsGetAgentEvalResult,
  evalsListAgentEvalResults,
} from '@nemo/sdk/generated/evals/evals-plugin-agent-eval-results-routes';
import type {
  AggregateRangeScore,
  AggregateRubricScore,
  AggregateScalarScore,
  AgentEvaluateJob,
  AgentEvaluateJobRequest,
  AgentEvalResult,
  AgentEvaluateJobsSortField,
  ResultFilter,
} from '@nemo/sdk/generated/evals/schema';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';

const PAGE_SIZE = 50;

/** Aggregate score — numeric range, rubric category distribution, or scalar value. */
export type AgentEvalAggregateScore =
  | AggregateRangeScore
  | AggregateRubricScore
  | AggregateScalarScore;

/** Re-export so callers continue to import AgentEvalResult from this module. */
export type { AgentEvalResult };

/** The agent name a job evaluated, read from its target (spec.target.agent.name).
 *  Strips only this job's own ``workspace/`` prefix so the result compares equal to a
 *  bare agent name; any other ``/`` in the name is left intact. */
export const agentNameForJob = (job: AgentEvaluateJob): string | null => {
  const target = job.spec?.target as
    | { kind?: string; agent?: { name?: string } }
    | null
    | undefined;
  if (!target || target.kind !== 'agent') return null;
  const name = target.agent?.name;
  if (typeof name !== 'string' || name.length === 0) return null;
  const prefix = job.workspace ? `${job.workspace}/` : '';
  return prefix && name.startsWith(prefix) ? name.slice(prefix.length) : name;
};

export const evalConfigName = (job: AgentEvaluateJob): string | null => {
  // Formerly ``spec.benchmark.eval_config_fileset``; AgentEvalSpec now carries
  // free-form ``labels`` (string map) after the evaluator OpenAPI regen.
  const name = job.spec?.labels?.eval_config_fileset;
  return typeof name === 'string' && name.length > 0 ? name : null;
};

export const fetchAgentEvalJobs = async (
  workspace: string,
  signal: AbortSignal
): Promise<AgentEvaluateJob[]> => {
  const all: AgentEvaluateJob[] = [];
  let page = 1;
  while (true) {
    const res = await evalsListAgentEvaluateJobs(
      workspace,
      { page, page_size: PAGE_SIZE, sort: '-created_at' as AgentEvaluateJobsSortField },
      signal
    );
    const batch = res?.data ?? [];
    all.push(...batch);
    if (batch.length < PAGE_SIZE) break;
    page++;
  }
  return all;
};

export const fetchAgentEvalJob = async (
  workspace: string,
  name: string,
  signal: AbortSignal
): Promise<AgentEvaluateJob | null> => {
  try {
    return await evalsGetAgentEvaluateJob(workspace, name, signal);
  } catch (err) {
    const e = err as { response?: { status?: number }; status?: number };
    if (e?.response?.status === 404 || e?.status === 404) return null;
    throw err;
  }
};

export const submitAgentEvalJob = async (
  workspace: string,
  request: AgentEvaluateJobRequest,
  signal?: AbortSignal
): Promise<AgentEvaluateJob> => evalsCreateAgentEvaluateJob(workspace, request, signal);

// ---------------------------------------------------------------------------
// Structured results (agent-eval-results record)
// ---------------------------------------------------------------------------

/** Flatten a result to its aggregate score rows (empty when none/absent). */
export const aggregateScoresOf = (result: AgentEvalResult | null): AgentEvalAggregateScore[] =>
  result?.scores?.scores ?? [];

export const fetchAgentEvalResult = async (
  workspace: string,
  name: string,
  signal: AbortSignal
): Promise<AgentEvalResult | null> => {
  try {
    return await evalsGetAgentEvalResult(workspace, name, signal);
  } catch (err) {
    const e = err as { response?: { status?: number }; status?: number };
    if (e?.response?.status === 404 || e?.status === 404) return null;
    throw err;
  }
};

const RESULTS_PAGE_MAX = 100;

export const fetchAgentEvalResultsForJobs = async (
  workspace: string,
  jobNames: string[],
  signal: AbortSignal
): Promise<Map<string, AgentEvalResult>> => {
  if (jobNames.length === 0) return new Map();
  const page = await evalsListAgentEvalResults(
    workspace,
    {
      page: 1,
      page_size: Math.min(jobNames.length, RESULTS_PAGE_MAX),
      filter: withOperators<ResultFilter>({ name: { $in: jobNames } }),
    },
    signal
  );
  return new Map((page?.data ?? []).map((result) => [result.name, result]));
};

/** One trial row from trials.jsonl — the agent's response to a task. */
export interface AgentEvalTrialRow {
  id: string;
  task_id: string;
  status: string;
  output?: { output_text?: string | null } | null;
  evidence?: unknown;
  metadata?: Record<string, unknown>;
}

/** One score row from scores.jsonl — a task's metric outputs + diagnostics. */
export interface AgentEvalScoreRow {
  id: string;
  task_id: string;
  trial_id: string;
  metric_type: string;
  status: string;
  outputs?: Array<{ name: string; value: number | string | null }>;
  diagnostics?: unknown[];
  metadata?: Record<string, unknown>;
}

/** One task row from tasks.jsonl — the evaluated input + ground truth. */
export interface AgentEvalTaskRow {
  id: string;
  intent?: string;
  inputs?: { instruction?: string | null };
  reference?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface AgentEvalBundle {
  tasks: AgentEvalTaskRow[];
  trials: AgentEvalTrialRow[];
  scores: AgentEvalScoreRow[];
}

/** Split a bundle_ref ("workspace/fileset#inner/path") into its parts. */
export const parseBundleRef = (
  bundleRef: string
): { fileset: string; innerPath: string } | null => {
  const [location, innerPath] = bundleRef.split('#');
  if (!location || !innerPath) return null;
  const fileset = location.includes('/')
    ? (location.split('/').slice(1).join('/') ?? '')
    : location;
  if (!fileset) return null;
  return { fileset, innerPath: innerPath.replace(/^\/+/, '') };
};

const downloadJsonl = async <T>(
  workspace: string,
  fileset: string,
  remotePath: string,
  signal: AbortSignal
): Promise<T[]> => {
  const blob = await filesDownloadFile(workspace, fileset, remotePath, signal);
  if (!blob) return [];
  const text = await blob.text();
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as T);
};

/** Loads the per-task bundle (tasks/trials/scores) for a completed job. Returns
 *  null when the bundle is not referenced or cannot be read (job not finished). */
export const fetchAgentEvalBundle = async (
  workspace: string,
  bundleRef: string | undefined,
  signal: AbortSignal
): Promise<AgentEvalBundle | null> => {
  if (!bundleRef) return null;
  const parsed = parseBundleRef(bundleRef);
  if (!parsed) return null;
  const { fileset, innerPath } = parsed;
  const at = (name: string): string => `${innerPath}/${name}`;
  try {
    const [tasks, trials, scores] = await Promise.all([
      downloadJsonl<AgentEvalTaskRow>(workspace, fileset, at('tasks.jsonl'), signal),
      downloadJsonl<AgentEvalTrialRow>(workspace, fileset, at('trials.jsonl'), signal),
      downloadJsonl<AgentEvalScoreRow>(workspace, fileset, at('scores.jsonl'), signal),
    ]);
    return { tasks, trials, scores };
  } catch (err) {
    const e = err as { response?: { status?: number }; status?: number };
    if (e?.response?.status === 404 || e?.status === 404) return null;
    throw err;
  }
};

/** A per-task row joining the task, its trial (response), and its score(s). */
export interface AgentEvalTaskDetail {
  taskId: string;
  trialId?: string;
  intent?: string;
  instruction?: string | null;
  reference?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  status: string;
  responseText?: string | null;
  scores: Array<{ name: string; value: number | string | null }>;
  diagnostics: unknown[];
}

/** Keeps a metric output usable by the score chip. Numbers pass through; rubric metrics
 *  emit a string label (e.g. `draft_quality.label` = "strong") which must survive rather
 *  than collapse to null and render as "not scored". Only a real NaN is unscored. */
const metricOutputValue = (value: number | string | null | undefined): number | string | null => {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string') return value.toLowerCase() === 'nan' ? null : value;
  return null;
};

const groupByTask = <T extends { task_id: string }>(rows: T[]): Map<string, T[]> => {
  const byTask = new Map<string, T[]>();
  for (const row of rows) byTask.set(row.task_id, [...(byTask.get(row.task_id) ?? []), row]);
  return byTask;
};

/** One task's row for a trial. Scores/diagnostics come from that same trial only, so they always
 *  match the responseText beside them. */
const taskDetail = (
  task: AgentEvalTaskRow,
  trial: AgentEvalTrialRow | undefined,
  scores: AgentEvalScoreRow[]
): AgentEvalTaskDetail => {
  const trialScores = scores.filter((s) => s.trial_id === trial?.id);
  return {
    taskId: task.id,
    trialId: trial?.id,
    intent: task.intent,
    instruction: task.inputs?.instruction ?? null,
    reference: task.reference,
    metadata: task.metadata,
    status: trial?.status ?? 'unknown',
    responseText: trial?.output?.output_text ?? null,
    scores: trialScores.flatMap((s) =>
      (s.outputs ?? []).map((o) => ({
        name: `${s.metric_type}.${o.name}`,
        value: metricOutputValue(o.value),
      }))
    ),
    diagnostics: trialScores.flatMap((s) => s.diagnostics ?? []),
  };
};

/** Join a bundle's tasks/trials/scores into one row per trial; a task with no trials keeps one row. */
export const joinBundleByTrial = (bundle: AgentEvalBundle | null): AgentEvalTaskDetail[] => {
  if (!bundle) return [];
  const trialsByTask = groupByTask(bundle.trials);
  const scoresByTask = groupByTask(bundle.scores);
  return bundle.tasks.flatMap((task) => {
    const trials = trialsByTask.get(task.id) ?? [undefined];
    return trials.map((trial) => taskDetail(task, trial, scoresByTask.get(task.id) ?? []));
  });
};

/** Join a bundle's tasks/trials/scores into one row per task, showing each task's last trial. */
export const joinBundleByTask = (bundle: AgentEvalBundle | null): AgentEvalTaskDetail[] => {
  if (!bundle) return [];
  const trialsByTask = groupByTask(bundle.trials);
  const scoresByTask = groupByTask(bundle.scores);
  return bundle.tasks.map((task) =>
    taskDetail(task, trialsByTask.get(task.id)?.at(-1), scoresByTask.get(task.id) ?? [])
  );
};
