// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Metric } from '@nemo/sdk/generated/evals/schema';
import {
  type ScoredResult,
  toMetricRuns,
  toMetricTrendSeries,
} from '@studio/api/metrics/metricScores';

const metric = {
  id: 'm1',
  name: 'answer-match',
  metric_type: 'exact-match',
  outputs: [
    { name: 'score', value_json_schema: {} },
    { name: 'partial', value_json_schema: {} },
  ],
} as unknown as Metric;

const result = (
  name: string,
  createdAt: string,
  scores: ScoredResult['scores']['scores']
): ScoredResult => ({
  name,
  job_id: name,
  kind: 'dataset',
  created_at: createdAt,
  target_name: 'llama',
  scores: { scores },
});

describe('toMetricRuns', () => {
  it('keeps runs that scored the metric, oldest first, reading scalar value and range mean', () => {
    const runs = toMetricRuns(metric, [
      result('newer', '2026-11-10T00:00:00Z', [
        { name: 'exact-match.score', nan_count: 0, score_type: 'scalar', value: 0.8 },
      ]),
      result('unrelated', '2026-11-09T00:00:00Z', [
        { name: 'bleu.score', nan_count: 0, score_type: 'scalar', value: 0.1 },
      ]),
      result('older', '2026-11-01T00:00:00Z', [
        { name: 'exact-match.score', nan_count: 0, score_type: 'range', mean: 0.5 },
        { name: 'exact-match.partial', nan_count: 0, score_type: 'range', mean: 0.7 },
      ]),
    ]);

    expect(runs.map((run) => run.name)).toEqual(['older', 'newer']);
    expect(runs[0].scores).toEqual({ score: 0.5, partial: 0.7 });
    expect(runs[1].scores).toEqual({ score: 0.8 });
  });

  it('drops scores that are not finite', () => {
    const runs = toMetricRuns(metric, [
      result('nan', '2026-11-01T00:00:00Z', [
        { name: 'exact-match.score', nan_count: 3, score_type: 'range', mean: Number.NaN },
      ]),
    ]);
    expect(runs).toEqual([]);
  });
});

describe('toMetricTrendSeries', () => {
  it('builds one series per scored output with the delta against the previous run', () => {
    const runs = toMetricRuns(metric, [
      result('a', '2026-11-01T00:00:00Z', [
        { name: 'exact-match.score', nan_count: 0, value: 0.5 },
        { name: 'exact-match.partial', nan_count: 0, value: 0.7 },
      ]),
      result('b', '2026-11-02T00:00:00Z', [
        { name: 'exact-match.score', nan_count: 0, value: 0.75 },
      ]),
    ]);

    const [score, partial] = toMetricTrendSeries(metric, runs);
    expect(score).toMatchObject({ id: 'score', value: 0.75, delta: 0.25 });
    expect(score.points.map((p) => p.value)).toEqual([0.5, 0.75]);
    expect(partial).toMatchObject({ id: 'partial', value: 0.7, delta: undefined });
  });

  it('returns no series when nothing scored the metric', () => {
    expect(toMetricTrendSeries(metric, [])).toEqual([]);
  });
});

describe('malformed responses', () => {
  it('skips results without a scores list and metrics without outputs', () => {
    const broken = { name: 'r', created_at: '2026-11-01T00:00:00Z', scores: {} } as ScoredResult;
    expect(toMetricRuns(metric, [broken])).toEqual([]);
    const noOutputs = { ...metric, outputs: undefined } as unknown as Metric;
    expect(toMetricTrendSeries(noOutputs, [])).toEqual([]);
  });
});
