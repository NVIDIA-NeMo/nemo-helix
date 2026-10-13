// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvaluateJob, AgentEvalResult } from '@nemo/sdk/generated/evals/schema';
import type { EvalComparisonEntry } from '@studio/components/dataViews/EvalComparisonTable/types';
import {
  baselineForComparisons,
  candidatesForComparisons,
  comparisonScoresForAgentEval,
  comparisonScoresForModelEval,
  comparisonsForEvalConfig,
  deltaFromBaseline,
  metricNamesForComparisons,
  normalizeScore,
  scoreForMetric,
} from '@studio/components/dataViews/EvalComparisonTable/utils';

const evaluations: EvalComparisonEntry[] = [
  {
    id: 'first',
    label: 'Baseline',
    createdAt: null,
    scores: [
      {
        name: 'correctness',
        count: 5,
        nan_count: 0,
        mean: 0.8,
        min: 0,
        max: 1,
        score_type: 'range',
      },
      {
        name: 'helpfulness',
        count: 5,
        nan_count: 0,
        mean: 0.6,
        min: 0,
        max: 1,
        score_type: 'range',
      },
    ],
  },
  {
    id: 'second',
    label: 'Candidate',
    createdAt: null,
    scores: [
      {
        name: 'correctness',
        count: 5,
        nan_count: 0,
        mean: 0.9,
        min: 0,
        max: 1,
        score_type: 'range',
      },
      { name: 'safety', count: 5, nan_count: 0, mean: 1, min: 0, max: 1, score_type: 'range' },
    ],
  },
];

describe('comparison score helpers', () => {
  it('builds a stable union of metric names and retrieves aggregate means', () => {
    expect(metricNamesForComparisons(evaluations)).toEqual([
      'correctness',
      'helpfulness',
      'safety',
    ]);
    expect(scoreForMetric(evaluations[1], 'correctness')).toBe(0.9);
    expect(scoreForMetric(evaluations[1], 'helpfulness')).toBeNull();
  });

  it('treats the first entry as the baseline and the rest as candidates', () => {
    expect(baselineForComparisons(evaluations)?.id).toBe('first');
    expect(candidatesForComparisons(evaluations).map((e) => e.id)).toEqual(['second']);
    expect(baselineForComparisons([])).toBeNull();
    expect(candidatesForComparisons([])).toEqual([]);
  });

  it('computes deltas against the baseline and leaves gaps null', () => {
    expect(deltaFromBaseline(evaluations[1], evaluations[0], 'correctness')).toEqual({
      value: 0.9,
      baselineValue: 0.8,
      difference: expect.closeTo(0.1, 10),
    });
    expect(deltaFromBaseline(evaluations[1], evaluations[0], 'helpfulness')).toEqual({
      value: null,
      baselineValue: 0.6,
      difference: null,
    });
    expect(deltaFromBaseline(evaluations[1], evaluations[0], 'safety')).toEqual({
      value: 1,
      baselineValue: null,
      difference: null,
    });
    expect(deltaFromBaseline(evaluations[1], null, 'correctness')).toEqual({
      value: 0.9,
      baselineValue: null,
      difference: null,
    });
  });

  it('normalizes scores against explicit bounds and clamps outliers', () => {
    expect(normalizeScore(75, { min: 50, max: 100 })).toBe(0.5);
    expect(normalizeScore(120, { min: 0, max: 100 })).toBe(1);
    expect(normalizeScore(null, { min: 0, max: 1 })).toBeNull();
    expect(normalizeScore(0.5, { min: 1, max: 1 })).toBeNull();
  });

  it('rejects non-finite score values rather than propagating them', () => {
    expect(normalizeScore(Number.NaN, { min: 0, max: 1 })).toBeNull();
    expect(normalizeScore(Number.POSITIVE_INFINITY, { min: 0, max: 1 })).toBeNull();
    expect(normalizeScore(Number.NEGATIVE_INFINITY, { min: 0, max: 1 })).toBeNull();
  });

  it('only includes jobs that share the requested eval config', () => {
    const jobs = [
      {
        id: 'one',
        name: 'baseline-run',
        workspace: 'default',
        created_at: '2026-01-01T00:00:00Z',
        spec: { labels: { eval_config_fileset: 'support-eval' } },
      },
      {
        id: 'two',
        name: 'other-run',
        workspace: 'default',
        spec: { labels: { eval_config_fileset: 'other-eval' } },
      },
    ] as unknown as AgentEvaluateJob[];

    const comparisons = comparisonsForEvalConfig(
      jobs,
      new Map([
        ['baseline-run', { name: 'baseline-run', scores: { scores: [] } }],
      ]) as unknown as Map<string, AgentEvalResult>,
      'support-eval'
    );

    expect(comparisons).toEqual([expect.objectContaining({ id: 'one', label: 'baseline-run' })]);
  });

  it('normalizes model-evaluation aggregate score artifacts', () => {
    expect(
      comparisonScoresForModelEval({
        exact_match: { mean: 0.86, min: 0, max: 1 },
        latency_s: { mean: 1.2 },
        missing_mean: { min: 0, max: 1 },
      })
    ).toEqual([
      { name: 'exact_match', mean: 0.86 },
      { name: 'latency_s', mean: 1.2 },
    ]);
  });

  it('normalizes an agent score with no mean to null', () => {
    expect(
      comparisonScoresForAgentEval([
        { name: 'rubric', count: 5, nan_count: 0, score_type: 'rubric' },
      ] as AgentEvalResult['scores']['scores'])
    ).toEqual([{ name: 'rubric', mean: null }]);
  });
});
