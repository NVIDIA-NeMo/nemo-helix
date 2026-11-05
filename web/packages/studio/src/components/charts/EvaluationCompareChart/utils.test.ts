// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SERIES_COLORS } from '@nemo/common/src/components/charts/tokens';
import type { EvaluationCompareResponse } from '@nemo/sdk/generated/platform/schema';
import {
  buildImprovementRows,
  buildMetricValueRows,
  describeComparison,
  describeImprovement,
  improvementPercent,
  symmetricAxisBound,
  toComparedEvaluations,
} from '@studio/components/charts/EvaluationCompareChart/utils';
import { deriveTrendMetrics } from '@studio/components/charts/ExperimentTrendChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

const row = (overrides: Partial<EvaluationRow> & { id: string }): EvaluationRow =>
  ({
    name: overrides.id,
    workspace: 'default',
    experiment_ids: ['exp'],
    dataset_name: 'ds',
    experiment_group_id: 'exp',
    ...overrides,
  }) as EvaluationRow;

const metricById = (rows: EvaluationRow[], id: string) => {
  const metric = deriveTrendMetrics(rows).find((m) => m.id === id);
  if (!metric) throw new Error(`no metric ${id}`);
  return metric;
};

const comparison = (overrides: Partial<EvaluationCompareResponse>): EvaluationCompareResponse =>
  ({
    workspace: 'default',
    baseline: 'base',
    candidate: 'cand',
    metric: 'evaluators.acc',
    higher_is_better: true,
    n_pairs: 2,
    better: 1,
    same: 1,
    worse: 0,
    mean_paired_difference: 0.25,
    interval_low: 0.1,
    interval_high: 0.4,
    p_candidate_better: 0.62,
    ...overrides,
  }) as EvaluationCompareResponse;

describe('toComparedEvaluations', () => {
  it('orders oldest first and assigns colors in that order', () => {
    const compared = toComparedEvaluations([
      row({ id: 'new', created_at: '2026-02-01T00:00:00Z' }),
      row({ id: 'old', created_at: '2026-01-01T00:00:00Z' }),
    ]);
    expect(compared.map(({ row: r }) => r.id)).toEqual(['old', 'new']);
    expect(compared.map(({ color }) => color)).toEqual([SERIES_COLORS[0], SERIES_COLORS[1]]);
  });
});

describe('improvementPercent', () => {
  const rows = [row({ id: 'a', aggregate_scores: { acc: { mean: 1 } } })];
  const score = metricById(rows, 'evaluators.acc');
  const cost = metricById(rows, 'cost_usd');

  it('is positive when a score rises', () => {
    expect(improvementPercent(score, 0.5, 0.6)).toBeCloseTo(20);
  });

  it('is positive when cost falls', () => {
    expect(improvementPercent(cost, 2, 1)).toBeCloseTo(50);
  });

  it('is undefined against a zero baseline', () => {
    expect(improvementPercent(score, 0, 0.5)).toBeUndefined();
  });

  it('is undefined when a side is missing', () => {
    expect(improvementPercent(score, undefined, 0.5)).toBeUndefined();
    expect(improvementPercent(score, 0.5, undefined)).toBeUndefined();
  });
});

describe('buildImprovementRows', () => {
  it('drops metrics no candidate can be compared on', () => {
    const rows = [
      row({ id: 'base', aggregate_scores: { acc: { mean: 0.5 } } }),
      row({ id: 'cand', aggregate_scores: { acc: { mean: 0.75 } }, cost_usd: { mean: 1 } }),
    ];
    const compared = toComparedEvaluations(rows);
    const valueRows = buildMetricValueRows(compared, deriveTrendMetrics(rows));
    const improvement = buildImprovementRows(valueRows, 'base', ['cand']);
    expect(improvement.map(({ metric }) => metric.id)).toEqual(['evaluators.acc']);
    expect(improvement[0]?.candidates.cand?.improvement).toBeCloseTo(50);
  });
});

describe('describeComparison', () => {
  it('summarizes the mean, interval, probability, and pair count', () => {
    expect(describeComparison(comparison({}))).toBe(
      'mean +0.25 (better) · 95% CI +0.1 to +0.4 · P(better) 62% · 2 paired test cases'
    );
  });

  it('gives no direction to a mean within the endpoint tie tolerance', () => {
    expect(describeComparison(comparison({ mean_paired_difference: 1e-12 }))).toBe(
      'mean +0 · 95% CI +0.1 to +0.4 · P(better) 62% · 2 paired test cases'
    );
  });

  it('keeps negative bounds signed and singularizes one pair', () => {
    expect(
      describeComparison(
        comparison({
          n_pairs: 1,
          mean_paired_difference: -0.5,
          interval_low: -1.25,
          interval_high: 0,
          p_candidate_better: 0.2,
        })
      )
    ).toBe('mean -0.5 (worse) · 95% CI -1.25 to 0 · P(better) 20% · 1 paired test case');
  });

  it('names a negative difference better for lower-is-better metrics', () => {
    expect(
      describeComparison(
        comparison({
          higher_is_better: false,
          mean_paired_difference: -0.05,
          interval_low: -0.12,
          interval_high: 0.02,
          p_candidate_better: 0.9,
        })
      )
    ).toBe('mean -0.05 (better) · 95% CI -0.12 to +0.02 · P(better) 90% · 2 paired test cases');
  });

  it('omits the pieces the response does not carry', () => {
    expect(
      describeComparison(
        comparison({
          mean_paired_difference: undefined,
          interval_low: undefined,
          interval_high: undefined,
          p_candidate_better: undefined,
        })
      )
    ).toBe('2 paired test cases');
  });

  it('says when nothing paired', () => {
    expect(
      describeComparison(
        comparison({
          n_pairs: 0,
          better: 0,
          same: 0,
          worse: 0,
          mean_paired_difference: undefined,
          interval_low: undefined,
          interval_high: undefined,
          p_candidate_better: undefined,
        })
      )
    ).toBe('no paired test cases');
  });
});

describe('describeImprovement', () => {
  it('names the direction instead of a sign', () => {
    expect(describeImprovement(42.64)).toBe('42.6% better');
    expect(describeImprovement(-9.3)).toBe('9.3% worse');
    expect(describeImprovement(0.01)).toBe('no change');
  });
});

describe('symmetricAxisBound', () => {
  it('rounds the largest magnitude up to a round bound', () => {
    expect(symmetricAxisBound([54.8, -12])).toBe(60);
    expect(symmetricAxisBound([81])).toBe(100);
    expect(symmetricAxisBound([-18])).toBe(20);
    expect(symmetricAxisBound([22])).toBe(25);
    expect(symmetricAxisBound([3.4])).toBe(4);
  });

  it('never collapses to zero', () => {
    expect(symmetricAxisBound([])).toBe(1);
    expect(symmetricAxisBound([0])).toBe(1);
  });
});
