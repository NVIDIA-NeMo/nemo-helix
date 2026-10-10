// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildTrendPoints,
  deriveTrendMetrics,
  type TrendMetric,
} from '@studio/components/charts/ExperimentTrendChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

// Only the fields the trend accessors read (name + created_at + the rollup means) are set; the rest
// of the rich EvaluationRow shape is irrelevant here, so build a minimal stand-in.
const row = (opts: {
  name: string;
  createdAt?: string;
  cost?: number;
  latency?: number;
  tokens?: number;
  durationSec?: string;
  evaluators?: Record<string, number>;
}): EvaluationRow =>
  ({
    name: opts.name,
    id: opts.name,
    created_at: opts.createdAt,
    metadata: opts.durationSec == null ? undefined : { eval_duration_sec: opts.durationSec },
    cost_usd: opts.cost == null ? undefined : { mean: opts.cost },
    latency_ms: opts.latency == null ? undefined : { mean: opts.latency },
    tokens: opts.tokens == null ? undefined : { mean: opts.tokens },
    aggregate_scores: opts.evaluators
      ? Object.fromEntries(Object.entries(opts.evaluators).map(([name, mean]) => [name, { mean }]))
      : undefined,
  }) as unknown as EvaluationRow;

const getMetric = (metrics: TrendMetric[], id: string): TrendMetric => {
  const metric = metrics.find((m) => m.id === id);
  if (!metric) throw new Error(`metric ${id} not found`);
  return metric;
};

describe('deriveTrendMetrics', () => {
  it('lists evaluators alphabetically, then the cost/latency/token rollups', () => {
    const metrics = deriveTrendMetrics([
      row({ name: 'b', evaluators: { report_style: 1 } }),
      row({ name: 'a', evaluators: { answers_question: 1, report_style: 1 } }),
    ]);
    expect(metrics.map((m) => m.id)).toEqual([
      'evaluators.answers_question',
      'evaluators.report_style',
      'duration_ms',
      'cost_usd',
      'latency_ms',
      'tokens',
    ]);
  });

  it('titles evaluator labels so the selector reads as prose', () => {
    const metrics = deriveTrendMetrics([row({ name: 'a', evaluators: { answers_question: 1 } })]);
    expect(getMetric(metrics, 'evaluators.answers_question').label).toBe('Answers Question');
  });

  it('keeps the full evaluator key when short labels would collide', () => {
    const metrics = deriveTrendMetrics([
      row({
        name: 'a',
        evaluators: { 'correctness.score': 1, 'tool_use.score': 1, 'style.tone': 1 },
      }),
    ]);
    expect(getMetric(metrics, 'evaluators.correctness.score').label).toBe('Correctness Score');
    expect(getMetric(metrics, 'evaluators.tool_use.score').label).toBe('Tool Use Score');
    expect(getMetric(metrics, 'evaluators.style.tone').label).toBe('Tone');
  });

  it('offers the rollups even when no evaluation carries an evaluator score', () => {
    const metrics = deriveTrendMetrics([row({ name: 'a', cost: 1 })]);
    expect(metrics.map((m) => m.id)).toEqual(['duration_ms', 'cost_usd', 'latency_ms', 'tokens']);
  });

  it('formats each metric in its own units', () => {
    const metrics = deriveTrendMetrics([row({ name: 'a', evaluators: { solved: 1 } })]);
    expect(getMetric(metrics, 'cost_usd').format(0.125)).toBe('$0.125');
    expect(getMetric(metrics, 'latency_ms').format(1234.6)).toBe('1,235 ms');
    expect(getMetric(metrics, 'tokens').format(16000)).toBe('16,000');
    expect(getMetric(metrics, 'evaluators.solved').format(0.60349)).toBe('0.603');
  });

  it('reads duration from the metadata the evaluator stamps, not from a rollup field', () => {
    const rows = [row({ name: 'a', createdAt: '2026-06-01T00:00:00Z', durationSec: '114.2' })];
    const [point] = buildTrendPoints(rows, getMetric(deriveTrendMetrics(rows), 'duration_ms'));
    expect(point.y).toBe(114200);
  });

  it('drops a run whose duration metadata is absent or unparseable', () => {
    // Metadata is free-form and only written on a successful publish, so both are ordinary states.
    const rows = [
      row({ name: 'no-metadata', createdAt: '2026-06-01T00:00:00Z' }),
      row({ name: 'blank', createdAt: '2026-06-02T00:00:00Z', durationSec: '' }),
      row({ name: 'junk', createdAt: '2026-06-03T00:00:00Z', durationSec: 'n/a' }),
      row({ name: 'good', createdAt: '2026-06-04T00:00:00Z', durationSec: '12' }),
    ];
    const points = buildTrendPoints(rows, getMetric(deriveTrendMetrics(rows), 'duration_ms'));
    expect(points.map((p) => p.name)).toEqual(['good']);
  });
});

describe('buildTrendPoints', () => {
  const metricsFor = (rows: EvaluationRow[]) => deriveTrendMetrics(rows);

  it('orders points oldest first regardless of input order', () => {
    const rows = [
      row({ name: 'mid', createdAt: '2026-06-05T00:00:00Z', evaluators: { solved: 0.7 } }),
      row({ name: 'new', createdAt: '2026-06-09T00:00:00Z', evaluators: { solved: 0.9 } }),
      row({ name: 'old', createdAt: '2026-06-01T00:00:00Z', evaluators: { solved: 0.5 } }),
    ];
    const points = buildTrendPoints(rows, getMetric(metricsFor(rows), 'evaluators.solved'));
    expect(points.map((p) => p.name)).toEqual(['old', 'mid', 'new']);
    expect(points.map((p) => p.y)).toEqual([0.5, 0.7, 0.9]);
  });

  it('places points at the evaluation timestamp so runs cluster by when they happened', () => {
    const rows = [row({ name: 'a', createdAt: '2026-06-01T12:00:00Z', cost: 0.5 })];
    const [point] = buildTrendPoints(rows, getMetric(metricsFor(rows), 'cost_usd'));
    expect(point.x).toBe(Date.parse('2026-06-01T12:00:00Z'));
  });

  it('drops rows with no parseable created_at, which cannot be placed on the axis', () => {
    const rows = [
      row({ name: 'undated', evaluators: { solved: 0.1 } }),
      row({ name: 'bad', createdAt: 'not-a-date', evaluators: { solved: 0.2 } }),
      row({ name: 'good', createdAt: '2026-06-01T00:00:00Z', evaluators: { solved: 0.8 } }),
    ];
    const points = buildTrendPoints(rows, getMetric(metricsFor(rows), 'evaluators.solved'));
    expect(points.map((p) => p.name)).toEqual(['good']);
  });

  it('drops rows missing the selected metric rather than plotting them as zero', () => {
    const rows = [
      row({ name: 'no-cost', createdAt: '2026-06-01T00:00:00Z', evaluators: { solved: 0.8 } }),
      row({ name: 'costed', createdAt: '2026-06-02T00:00:00Z', cost: 0.25 }),
    ];
    const points = buildTrendPoints(rows, getMetric(metricsFor(rows), 'cost_usd'));
    expect(points.map((p) => p.name)).toEqual(['costed']);
  });

  it('returns nothing when no row carries the metric, so the chart can show its empty state', () => {
    const rows = [row({ name: 'a', createdAt: '2026-06-01T00:00:00Z', evaluators: { solved: 1 } })];
    expect(buildTrendPoints(rows, getMetric(metricsFor(rows), 'tokens'))).toEqual([]);
  });
});
