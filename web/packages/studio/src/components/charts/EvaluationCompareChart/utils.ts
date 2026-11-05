// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SERIES_COLORS } from '@nemo/common/src/components/charts/tokens';
import type { EvaluationCompareResponse } from '@nemo/sdk/generated/platform/schema';
import type { TrendMetric } from '@studio/components/charts/ExperimentTrendChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

export const MIN_COMPARED_EVALUATIONS = 2;
export const MAX_COMPARED_EVALUATIONS = SERIES_COLORS.length;

const EVALUATOR_METRIC_PREFIX = 'evaluators.';

export const isEvaluatorMetric = (metric: TrendMetric): boolean =>
  metric.id.startsWith(EVALUATOR_METRIC_PREFIX);

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
 * Win/tie/loss counts of a candidate over the baseline across the paired test cases, as
 * computed by the compare endpoint.
 */
export interface HeadToHead {
  readonly better: number;
  readonly same: number;
  readonly worse: number;
}

const formatSignedDifference = (value: number): string =>
  `${value > 0 ? '+' : ''}${value.toLocaleString(undefined, { maximumFractionDigits: 3 })}`;

/** The direction the endpoint reports: for cost and latency a negative difference is better. */
const favorsCandidate = (compare: EvaluationCompareResponse, difference: number): boolean =>
  compare.higher_is_better ? difference > 0 : difference < 0;

/**
 * One line summarizing the endpoint's paired comparison for one candidate: the mean paired
 * difference (signed candidate − baseline, with better/worse per the metric's direction),
 * its 95% bootstrap interval, and P(candidate better), for the one selected metric.
 */
export function describeComparison(compare: EvaluationCompareResponse): string {
  if (compare.n_pairs === 0) return 'no paired test cases';
  const parts: string[] = [];
  if (compare.mean_paired_difference != null) {
    const mean = compare.mean_paired_difference;
    const direction = mean === 0 ? '' : favorsCandidate(compare, mean) ? ' (better)' : ' (worse)';
    parts.push(`mean ${formatSignedDifference(mean)}${direction}`);
  }
  if (compare.interval_low != null && compare.interval_high != null) {
    parts.push(
      `95% CI ${formatSignedDifference(compare.interval_low)} to ${formatSignedDifference(compare.interval_high)}`
    );
  }
  if (compare.p_candidate_better != null) {
    parts.push(`P(better) ${Math.round(compare.p_candidate_better * 100)}%`);
  }
  parts.push(`${compare.n_pairs} paired test case${compare.n_pairs === 1 ? '' : 's'}`);
  return parts.join(' · ');
}

const NICE_STEPS = [1, 2, 2.5, 4, 5, 6, 8, 10];

/** Smallest round bound (…, 10, 20, 25, 40, 50, 60, 80, 100, …) covering every value, so ±half ticks stay round. */
export function symmetricAxisBound(values: readonly number[]): number {
  const largest = Math.max(1, ...values.map(Math.abs));
  const magnitude = 10 ** Math.floor(Math.log10(largest));
  const step = NICE_STEPS.find((s) => s * magnitude >= largest) ?? 10;
  return step * magnitude;
}

export const CATEGORY_AXIS_WIDTH = 140;

export const truncateCategoryLabel = (label: string): string =>
  label.length > 20 ? `${label.slice(0, 19)}…` : label;

export const formatImprovement = (value: number): string =>
  `${value > 0 ? '+' : ''}${value.toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;

/** Spelled out because a positive improvement can be a value going down (cost, latency). */
export const describeImprovement = (value: number): string => {
  const magnitude = Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (magnitude === '0') return 'no change';
  return `${magnitude}% ${value > 0 ? 'better' : 'worse'}`;
};
