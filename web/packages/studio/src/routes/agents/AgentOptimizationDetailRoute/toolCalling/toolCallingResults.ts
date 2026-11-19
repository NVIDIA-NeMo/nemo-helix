// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import {
  downloadText,
  locateJobResultFiles,
  siblingPath,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';
import { z } from 'zod';

/** Written by the nemo-tool-calling plugin's `tool-calling` strategy; see its CONTRACT.md. */
export const TOOL_CALLING_RESULT_FILE = 'tool-calling-result.json';

const rate = z.number().min(0).max(1);

export const variantIdSchema = z.enum(['baseline', 'config-fix', 'schema-fix']);
export const findingSourceSchema = z.enum(['schema-lint', 'failure-analysis']);
export const findingSeveritySchema = z.enum(['high', 'medium', 'low']);
export const suggestionKindSchema = z.enum(['agent-config', 'tool-schema']);
export const callSignalSchema = z.enum(['tool_error', 'schema_invalid', 'error_text']);

const signalCountsSchema = z.object({
  tool_error: z.number().default(0),
  schema_invalid: z.number().default(0),
  error_text: z.number().default(0),
  retry_loop_runs: z.number().default(0),
});

const variantMetricsSchema = z.object({
  signals: signalCountsSchema.optional(),
  runs: z.number(),
  runs_with_errors: z.number(),
  tool_calls: z.number(),
  rejected_calls: z.number(),
  tool_error_rate: rate,
  first_call_accepted_rate: rate,
  task_success_rate: rate,
  avg_calls_per_run: z.number(),
});

const rejectedCallExampleSchema = z.object({
  prompt_id: z.string(),
  tool: z.string(),
  arguments_excerpt: z.string(),
  rejection: z.string(),
  signals: z.array(callSignalSchema).default([]),
  schema_errors: z.array(z.string()).default([]),
});

const variantSchema = z.object({
  id: variantIdSchema,
  label: z.string(),
  description: z.string().optional(),
  metrics: variantMetricsSchema,
  rejection_reasons: z.record(z.number()).default({}),
  per_prompt: z.record(variantMetricsSchema).optional(),
  examples: z.array(rejectedCallExampleSchema).default([]),
});

const findingSchema = z.object({
  id: z.string(),
  source: findingSourceSchema,
  severity: findingSeveritySchema,
  tool: z.string().nullish(),
  title: z.string(),
  detail: z.string().default(''),
  evidence: z.array(z.string()).default([]),
});

const suggestionSchema = z.object({
  id: z.string(),
  kind: suggestionKindSchema,
  title: z.string(),
  rationale: z.string().default(''),
  applies_automatically: z.boolean(),
  artifact: z.string(),
  diff_artifact: z.string().nullish(),
  verified_by_variant: variantIdSchema.nullish(),
});

export const toolCallingResultSchema = z.object({
  version: z.number(),
  agent: z.string(),
  analyzer_model: z.string().nullish(),
  dataset: z.object({ prompts: z.number(), runs_per_prompt: z.number() }).nullish(),
  variants: z.array(variantSchema).min(1),
  findings: z.array(findingSchema).default([]),
  suggestions: z.array(suggestionSchema).default([]),
  recommended_variant: variantIdSchema.exclude(['baseline']).nullish(),
  recommendation: z.string().nullish(),
});

export type VariantId = z.infer<typeof variantIdSchema>;
export type VariantMetrics = z.infer<typeof variantMetricsSchema>;
export type SignalCounts = z.infer<typeof signalCountsSchema>;
export type CallSignal = z.infer<typeof callSignalSchema>;
export type ToolCallingVariant = z.infer<typeof variantSchema>;
export type RejectedCallExample = z.infer<typeof rejectedCallExampleSchema>;
export type ToolCallingFinding = z.infer<typeof findingSchema>;
export type ToolCallingSuggestion = z.infer<typeof suggestionSchema>;
export type ToolCallingResult = z.infer<typeof toolCallingResultSchema>;

export interface ResultLocation {
  workspace: string;
  fileset: string;
  /** Fileset path of the result file; artifacts it names resolve beside it. */
  indexPath: string;
}

export interface ToolCallingResults {
  result: ToolCallingResult;
  location: ResultLocation;
}

export const parseToolCallingResult = (text: string): ToolCallingResult => {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error(`${TOOL_CALLING_RESULT_FILE} is not valid JSON.`);
  }
  const parsed = toolCallingResultSchema.safeParse(raw);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const where = issue?.path.length ? ` at ${issue.path.join('.')}` : '';
    throw new Error(
      `${TOOL_CALLING_RESULT_FILE} does not match the expected shape${where}: ${issue?.message ?? 'invalid'}.`
    );
  }
  return parsed.data;
};

/** Read the run's result summary; null when the job saved none. */
export const fetchToolCallingResults = async (
  workspace: string,
  jobName: string,
  signal?: AbortSignal
): Promise<ToolCallingResults | null> => {
  const located = await locateJobResultFiles(
    workspace,
    jobName,
    [TOOL_CALLING_RESULT_FILE],
    signal
  );
  const indexPath = located?.paths[TOOL_CALLING_RESULT_FILE];
  if (!located || !indexPath) {
    return null;
  }

  const text = await downloadText(located.workspace, located.fileset, indexPath, signal);
  if (!text) {
    return null;
  }
  return {
    result: parseToolCallingResult(text),
    location: { workspace: located.workspace, fileset: located.fileset, indexPath },
  };
};

export const fetchArtifactText = async (
  location: ResultLocation,
  artifact: string,
  signal?: AbortSignal
): Promise<string> => {
  const path = siblingPath(location.indexPath, artifact);
  const text = await downloadText(location.workspace, location.fileset, path, signal);
  if (text === null) {
    throw new Error(`The artifact "${artifact}" is missing.`);
  }
  return text;
};

export const fetchArtifactBlob = async (
  location: ResultLocation,
  artifact: string
): Promise<Blob> => {
  const blob = await filesDownloadFile(
    location.workspace,
    location.fileset,
    siblingPath(location.indexPath, artifact)
  );
  if (!blob) {
    throw new Error(`The artifact "${artifact}" is missing.`);
  }
  return blob;
};

export type MetricFormat = 'rate' | 'decimal' | 'count';

export type ComparedMetric = keyof Pick<
  VariantMetrics,
  | 'tool_error_rate'
  | 'first_call_accepted_rate'
  | 'task_success_rate'
  | 'avg_calls_per_run'
  | 'runs_with_errors'
>;

export interface MetricRow {
  key: ComparedMetric;
  label: string;
  format: MetricFormat;
  lowerIsBetter: boolean;
}

export const METRIC_ROWS: readonly MetricRow[] = [
  { key: 'tool_error_rate', label: 'Tool error rate', format: 'rate', lowerIsBetter: true },
  {
    key: 'first_call_accepted_rate',
    label: 'First call accepted',
    format: 'rate',
    lowerIsBetter: false,
  },
  { key: 'task_success_rate', label: 'Task success rate', format: 'rate', lowerIsBetter: false },
  { key: 'avg_calls_per_run', label: 'Avg calls per run', format: 'decimal', lowerIsBetter: true },
  { key: 'runs_with_errors', label: 'Runs with errors', format: 'count', lowerIsBetter: true },
];

export const METRIC_DESCRIPTIONS: Record<MetricRow['key'], string> = {
  tool_error_rate:
    'Share of tool calls that were bad: rejected by the tool, invalid against its schema, or error-shaped. Lower is better.',
  first_call_accepted_rate:
    'Share of runs whose very first tool call was accepted, with no retry needed.',
  task_success_rate:
    'Share of runs whose last tool call was accepted, meaning the agent got there eventually.',
  avg_calls_per_run: 'Tool calls per run. Far above 1 means the agent is retrying.',
  runs_with_errors: 'Runs that ended in a step-limit, timeout or transport error.',
};

export const HEADLINE_METRIC = METRIC_ROWS[0];

export const CALL_SIGNAL_LABEL: Record<CallSignal, string> = {
  tool_error: 'Tool error',
  schema_invalid: 'Schema invalid',
  error_text: 'Error text (heuristic)',
};

export interface SignalRow {
  key: keyof SignalCounts;
  label: string;
}

export const SIGNAL_ROWS: readonly SignalRow[] = [
  ...callSignalSchema.options.map((key) => ({ key, label: CALL_SIGNAL_LABEL[key] })),
  { key: 'retry_loop_runs', label: 'Runs with a retry loop' },
];

export const SIGNAL_DESCRIPTIONS: Record<SignalRow['key'], string> = {
  tool_error: 'The tool rejected the call.',
  schema_invalid:
    "The arguments don't match the tool's published schema. The platform checks this itself.",
  error_text: "The result looks like an error, but the tool didn't say so explicitly.",
  retry_loop_runs:
    'A run called the same tool again right after a bad call, or used more calls than the budget.',
};

export const formatMetric = (value: number, format: MetricFormat): string => {
  if (format === 'rate') {
    return `${Math.round(value * 100)}%`;
  }
  if (format === 'decimal') {
    return value.toFixed(1);
  }
  return String(value);
};

const MINUS = '−';

export const formatDelta = (delta: number, format: MetricFormat): string => {
  const sign = delta > 0 ? '+' : delta < 0 ? MINUS : '';
  const magnitude = Math.abs(delta);
  if (format === 'rate') {
    return `${sign}${Math.round(magnitude * 100)} pts`;
  }
  if (format === 'decimal') {
    return `${sign}${magnitude.toFixed(1)}`;
  }
  return `${sign}${magnitude}`;
};

export type DeltaDirection = 'better' | 'worse' | 'same';

export const deltaDirection = (delta: number, lowerIsBetter: boolean): DeltaDirection => {
  if (Math.abs(delta) < 1e-9) {
    return 'same';
  }
  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  return improved ? 'better' : 'worse';
};

/** Every rejection reason any variant saw, most frequent in the baseline first. */
export const rejectionReasonRows = (
  variants: readonly ToolCallingVariant[]
): { reason: string; counts: Partial<Record<VariantId, number>> }[] => {
  const byReason = new Map<string, Partial<Record<VariantId, number>>>();
  for (const variant of variants) {
    for (const [reason, count] of Object.entries(variant.rejection_reasons)) {
      const counts = byReason.get(reason) ?? {};
      counts[variant.id] = count;
      byReason.set(reason, counts);
    }
  }
  const total = (counts: Partial<Record<VariantId, number>>) =>
    Object.values(counts).reduce((sum, count) => sum + (count ?? 0), 0);
  return [...byReason.entries()]
    .map(([reason, counts]) => ({ reason, counts }))
    .sort(
      (a, b) =>
        (b.counts.baseline ?? 0) - (a.counts.baseline ?? 0) ||
        total(b.counts) - total(a.counts) ||
        a.reason.localeCompare(b.reason)
    );
};
