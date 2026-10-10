// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SERIES_COLORS } from '@nemo/common/src/components/charts/tokens';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import type { TrendMetric } from '@studio/components/charts/ExperimentTrendChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

export const MIN_COMPARED_EVALUATIONS = 2;
export const MAX_COMPARED_EVALUATIONS = SERIES_COLORS.length;

const EVALUATOR_METRIC_PREFIX = 'evaluators.';

export const isEvaluatorMetric = (metric: TrendMetric): boolean =>
  metric.id.startsWith(EVALUATOR_METRIC_PREFIX);

export const evaluatorNameOf = (metric: TrendMetric): string =>
  metric.id.slice(EVALUATOR_METRIC_PREFIX.length);

/** Scores go up when things get better; cost, latency, tokens and duration go down. */
export const higherIsBetter = isEvaluatorMetric;

export interface ComparedEvaluation {
  readonly row: EvaluationRow;
  readonly color: string;
}

/** Oldest first, so colors follow the run rather than the order rows were ticked in. */
export function toComparedEvaluations(rows: readonly EvaluationRow[]): ComparedEvaluation[] {
  return [...rows]
    .sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
    .slice(0, MAX_COMPARED_EVALUATIONS)
    .map((row, index) => ({ row, color: SERIES_COLORS[index] }));
}

const finite = (value: number | null | undefined): number | undefined =>
  value != null && Number.isFinite(value) ? value : undefined;

export interface MetricValuesRow {
  readonly metric: TrendMetric;
  readonly values: Readonly<Record<string, number | undefined>>;
}

/** One row per metric that at least one compared evaluation has a value for. */
export function buildMetricValueRows(
  evaluations: readonly ComparedEvaluation[],
  metrics: readonly TrendMetric[]
): MetricValuesRow[] {
  return metrics
    .map((metric) => ({
      metric,
      values: Object.fromEntries(
        evaluations.map(({ row }) => [row.id, finite(metric.accessor(row))])
      ),
    }))
    .filter(({ values }) => Object.values(values).some((value) => value !== undefined));
}

/**
 * Relative change against the baseline, signed so positive always means better. Undefined when
 * either side is missing or the baseline is zero, where a ratio has no meaning.
 */
export function improvementPercent(
  metric: TrendMetric,
  baseline: number | undefined,
  candidate: number | undefined
): number | undefined {
  if (baseline === undefined || candidate === undefined || baseline === 0) return undefined;
  const change = ((candidate - baseline) / Math.abs(baseline)) * 100;
  return higherIsBetter(metric) ? change : -change;
}

export interface ImprovementRow {
  readonly metric: TrendMetric;
  readonly baseline: number | undefined;
  readonly candidates: Readonly<Record<string, { value?: number; improvement?: number }>>;
}

export function buildImprovementRows(
  valueRows: readonly MetricValuesRow[],
  baselineId: string,
  candidateIds: readonly string[]
): ImprovementRow[] {
  return valueRows
    .map(({ metric, values }) => {
      const baseline = values[baselineId];
      const candidates = Object.fromEntries(
        candidateIds.map((id) => [
          id,
          { value: values[id], improvement: improvementPercent(metric, baseline, values[id]) },
        ])
      );
      return { metric, baseline, candidates };
    })
    .filter(({ candidates }) =>
      Object.values(candidates).some(({ improvement }) => improvement !== undefined)
    );
}

/**
 * Mean score per test case for one evaluator. A failed evaluator result counts as 0, matching
 * how the evaluation-level mean treats it. Sessions without a test case name cannot be paired
 * across runs, so they are left out.
 */
export function meanScoreByTestCase(
  sessions: readonly EvaluationSessionResponse[],
  evaluatorName: string
): Map<string, number> {
  const totals = new Map<string, { sum: number; count: number }>();
  for (const session of sessions) {
    const testCase = session.test_case_name;
    if (!testCase) continue;
    const failed = session.failed_evaluators?.includes(evaluatorName) ?? false;
    const score = failed ? 0 : finite(session.evaluator_scores?.[evaluatorName]);
    if (score === undefined) continue;
    const total = totals.get(testCase) ?? { sum: 0, count: 0 };
    totals.set(testCase, { sum: total.sum + score, count: total.count + 1 });
  }
  return new Map([...totals].map(([testCase, { sum, count }]) => [testCase, sum / count]));
}

export interface HeadToHead {
  readonly better: number;
  readonly same: number;
  readonly worse: number;
}

const SAME_SCORE_TOLERANCE = 1e-9;

/** Win/tie/loss of the candidate over the baseline across the test cases both runs scored. */
export function headToHead(
  baseline: ReadonlyMap<string, number>,
  candidate: ReadonlyMap<string, number>
): HeadToHead {
  let better = 0;
  let same = 0;
  let worse = 0;
  for (const [testCase, candidateScore] of candidate) {
    const baselineScore = baseline.get(testCase);
    if (baselineScore === undefined) continue;
    const diff = candidateScore - baselineScore;
    if (Math.abs(diff) <= SAME_SCORE_TOLERANCE) same += 1;
    else if (diff > 0) better += 1;
    else worse += 1;
  }
  return { better, same, worse };
}

const NICE_STEPS = [1, 2, 2.5, 4, 5, 6, 8, 10];

/** Smallest round bound (…, 10, 20, 25, 40, 50, 60, 80, 100, …) covering every value, so ±half ticks stay round. */
export function symmetricAxisBound(values: readonly number[]): number {
  const largest = Math.max(1, ...values.map(Math.abs));
  const magnitude = 10 ** Math.floor(Math.log10(largest));
  const step = NICE_STEPS.find((s) => s * magnitude >= largest) ?? 10;
  return step * magnitude;
}

export const formatImprovement = (value: number): string =>
  `${value > 0 ? '+' : ''}${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;

/** Spelled out because a positive improvement can be a value going down (cost, latency). */
export const describeImprovement = (value: number): string => {
  const magnitude = Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (magnitude === '0') return 'no change';
  return `${magnitude}% ${value > 0 ? 'better' : 'worse'}`;
};
