// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvaluateJob } from '@nemo/sdk/generated/evals/schema';
import {
  agentNameForJob,
  aggregateScoresOf,
  evalConfigName,
  type AgentEvalAggregateScore,
  type AgentEvalResult,
} from '@studio/api/evaluation/agent-evaluations';
import type {
  EvalComparisonEntry,
  EvalComparisonScore,
  ComparisonMetricBounds,
  ComparisonMetricDelta,
} from '@studio/components/dataViews/EvalComparisonTable/types';

/** Creates comparison rows from evaluator API responses and keeps only runs tied to one
 * persisted eval-config fileset. `resultsByJobName` is the map returned by
 * `fetchAgentEvalResultsForJobs`. */
export const comparisonsForEvalConfig = (
  jobs: readonly AgentEvaluateJob[],
  resultsByJobName: ReadonlyMap<string, AgentEvalResult>,
  configName: string
): EvalComparisonEntry[] =>
  jobs.flatMap((job) => {
    if (evalConfigName(job) !== configName || !job.name) return [];
    const agentName = agentNameForJob(job);
    return [
      {
        id: job.id || job.name,
        label: agentName ?? job.name,
        createdAt: job.created_at ?? null,
        scores: comparisonScoresForAgentEval(
          aggregateScoresOf(resultsByJobName.get(job.name) ?? null)
        ),
      },
    ];
  });

const aggregateScoreValue = (score: AgentEvalAggregateScore): number | null => {
  const value = score.mean;
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

/** Normalizes optional agent-evaluation values to the comparison table's explicit null value. */
export const comparisonScoresForAgentEval = (
  scores: readonly AgentEvalAggregateScore[]
): EvalComparisonScore[] =>
  scores.map((score) => ({ name: score.name, mean: aggregateScoreValue(score) }));

/** Selects the aggregate mean for every metric in the model-evaluation result artifact.
 * Model results use a record keyed by metric name, whereas agent results are already an
 * array; both become `EvalComparisonScore` before reaching the table. */
export const comparisonScoresForModelEval = (
  scores: Readonly<Record<string, Readonly<Record<string, number>>>> | null | undefined
): EvalComparisonScore[] =>
  Object.entries(scores ?? {}).flatMap(([name, aggregate]) => {
    const mean = aggregate.mean;
    return typeof mean === 'number' && Number.isFinite(mean) ? [{ name, mean }] : [];
  });

export const metricNamesForComparisons = (entries: readonly EvalComparisonEntry[]): string[] =>
  Array.from(new Set(entries.flatMap((entry) => entry.scores.map((score) => score.name))));

export const scoreForMetric = (entry: EvalComparisonEntry, metricName: string): number | null => {
  const score = entry.scores.find((candidate) => candidate.name === metricName)?.mean;
  return typeof score === 'number' && Number.isFinite(score) ? score : null;
};

/** The run every other run is compared against. Callers control this by ordering the list. */
export const baselineForComparisons = (
  entries: readonly EvalComparisonEntry[]
): EvalComparisonEntry | null => entries[0] ?? null;

/** Every run after the baseline, in the order supplied. */
export const candidatesForComparisons = (
  entries: readonly EvalComparisonEntry[]
): EvalComparisonEntry[] => entries.slice(1);

export const deltaFromBaseline = (
  entry: EvalComparisonEntry,
  baseline: EvalComparisonEntry | null,
  metricName: string
): ComparisonMetricDelta => {
  const value = scoreForMetric(entry, metricName);
  const baselineValue = baseline ? scoreForMetric(baseline, metricName) : null;
  return {
    value,
    baselineValue,
    difference: value !== null && baselineValue !== null ? value - baselineValue : null,
  };
};

/** Converts a metric value into a 0–1 chart value. Missing and non-finite values remain null so
 * callers can make their missing-data policy explicit. */
export const normalizeScore = (
  value: number | null,
  bounds: ComparisonMetricBounds | undefined
): number | null => {
  if (value === null || !Number.isFinite(value)) return null;
  const min = bounds?.min ?? 0;
  const max = bounds?.max ?? 1;
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
  return Math.min(1, Math.max(0, (value - min) / (max - min)));
};
