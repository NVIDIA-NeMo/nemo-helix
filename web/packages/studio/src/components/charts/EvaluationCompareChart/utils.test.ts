// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SERIES_COLORS } from '@nemo/common/src/components/charts/tokens';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import {
  buildImprovementRows,
  buildMetricValueRows,
  describeImprovement,
  headToHead,
  improvementPercent,
  meanScoreByTestCase,
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

const session = (
  testCase: string | undefined,
  scores: Record<string, number>,
  failed: string[] = []
): EvaluationSessionResponse =>
  ({
    workspace: 'default',
    evaluation_name: 'e',
    session_id: Math.random().toString(),
    test_case_name: testCase,
    trace_id: 't',
    root_span_id: 'r',
    started_at: '2026-01-01T00:00:00Z',
    status: 'success',
    evaluator_scores: scores,
    failed_evaluators: failed,
  }) as EvaluationSessionResponse;

const metricById = (rows: EvaluationRow[], id: string) => {
  const metric = deriveTrendMetrics(rows).find((m) => m.id === id);
  if (!metric) throw new Error(`no metric ${id}`);
  return metric;
};

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

describe('meanScoreByTestCase', () => {
  it('averages repeated attempts and counts a failed evaluator as zero', () => {
    const scores = meanScoreByTestCase(
      [session('tc1', { acc: 1 }), session('tc1', {}, ['acc']), session('tc2', { acc: 0.5 })],
      'acc'
    );
    expect(Object.fromEntries(scores)).toEqual({ tc1: 0.5, tc2: 0.5 });
  });

  it('skips sessions without a test case name', () => {
    expect(meanScoreByTestCase([session(undefined, { acc: 1 })], 'acc').size).toBe(0);
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

describe('headToHead', () => {
  it('counts only test cases both runs scored', () => {
    const baseline = new Map([
      ['a', 0.5],
      ['b', 0.5],
      ['c', 0.5],
      ['only-baseline', 1],
    ]);
    const candidate = new Map([
      ['a', 0.9],
      ['b', 0.5],
      ['c', 0.1],
      ['only-candidate', 1],
    ]);
    expect(headToHead(baseline, candidate)).toEqual({ better: 1, same: 1, worse: 1 });
  });
});
