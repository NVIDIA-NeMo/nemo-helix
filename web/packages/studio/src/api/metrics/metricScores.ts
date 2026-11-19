// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AgentEvalResult,
  AggregatedMetricResult,
  EvaluateResult,
  Metric,
} from '@nemo/sdk/generated/evals/schema';
import type { MetricTrendSeries } from '@studio/components/charts/MetricTrend';

/** The fields of an eval result (dataset or agent) the trend reads. */
export type ScoredResult = Pick<
  EvaluateResult | AgentEvalResult,
  'name' | 'job_id' | 'created_at' | 'target_kind' | 'target_name' | 'scores'
> & {
  /** Which results list it came from; each kind has its own detail page. */
  kind: 'dataset' | 'agent';
};

type AggregateScore = AggregatedMetricResult['scores'][number];

export interface MetricRun {
  name: string;
  kind: ScoredResult['kind'];
  jobId: string;
  createdAt: string;
  targetKind?: string;
  targetName?: string;
  /** Output name to aggregate score, for the outputs this run scored. */
  scores: Record<string, number>;
}

/**
 * Aggregates are named `<metric_type>.<output>` by the evaluator, not by the stored metric's
 * name, so two stored metrics of the same type share one score history.
 */
const scoreName = (metric: Metric, output: string) => `${metric.metric_type}.${output}`;

const scoreValue = (score: AggregateScore): number | undefined => {
  const value = 'value' in score ? score.value : score.mean;
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
};

/**
 * The generated types mark these required, but older deployments return results without the
 * `scores.scores` list and metrics without `outputs`, so read both defensively.
 */
export const metricOutputNames = (metric: Metric): string[] =>
  Array.isArray(metric.outputs) ? metric.outputs.map((output) => output.name) : [];

const aggregateScores = (result: ScoredResult): AggregateScore[] => {
  const scores: unknown = result.scores?.scores;
  return Array.isArray(scores) ? (scores as AggregateScore[]) : [];
};

const pointLabel = (at: string) =>
  new Date(at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

/** Runs that scored this metric, oldest first. */
export const toMetricRuns = (metric: Metric, results: ScoredResult[]): MetricRun[] =>
  results
    .flatMap((result) => {
      const byName = new Map(aggregateScores(result).map((score) => [score.name, score]));
      const scores = Object.fromEntries(
        metricOutputNames(metric).flatMap((name) => {
          const score = byName.get(scoreName(metric, name));
          const value = score && scoreValue(score);
          return value === undefined ? [] : [[name, value]];
        })
      );
      return Object.keys(scores).length > 0 && Number.isFinite(Date.parse(result.created_at))
        ? [
            {
              name: result.name,
              kind: result.kind,
              jobId: result.job_id,
              createdAt: result.created_at,
              targetKind: result.target_kind,
              targetName: result.target_name,
              scores,
            },
          ]
        : [];
    })
    .sort((a, b) => Date.parse(a.createdAt) - Date.parse(b.createdAt));

/** One trend series per output, with the delta measured against the previous run. */
export const toMetricTrendSeries = (metric: Metric, runs: MetricRun[]): MetricTrendSeries[] =>
  metricOutputNames(metric).flatMap((name) => {
    const points = runs.flatMap((run) =>
      name in run.scores ? [{ label: pointLabel(run.createdAt), value: run.scores[name] }] : []
    );
    const latest = points.at(-1);
    if (!latest) return [];
    const previous = points.at(-2);
    return [
      {
        id: name,
        label: name,
        value: latest.value,
        delta: previous && latest.value - previous.value,
        points,
      },
    ];
  });
