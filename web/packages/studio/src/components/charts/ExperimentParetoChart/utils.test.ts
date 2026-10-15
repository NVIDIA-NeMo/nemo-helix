// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildParetoPoints,
  deriveParetoMetrics,
  type ParetoMetric,
  resolveParetoAxes,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

// Only the fields the Pareto accessors read (name + cost/latency/tokens/evaluator rollup means) are set;
// the rest of the rich EvaluationRow shape is irrelevant here, so build a minimal stand-in.
const point = (opts: {
  name: string;
  cost?: number;
  latency?: number;
  tokens?: number;
  evaluators?: Record<string, number>;
}): EvaluationRow =>
  ({
    name: opts.name,
    id: opts.name,
    cost_usd: opts.cost == null ? undefined : { mean: opts.cost },
    latency_ms: opts.latency == null ? undefined : { mean: opts.latency },
    tokens: opts.tokens == null ? undefined : { mean: opts.tokens },
    aggregate_scores: opts.evaluators
      ? Object.fromEntries(Object.entries(opts.evaluators).map(([name, mean]) => [name, { mean }]))
      : undefined,
  }) as unknown as EvaluationRow;

const getMetric = (metrics: ParetoMetric[], id: string): ParetoMetric => {
  const metric = metrics.find((m) => m.id === id);
  if (!metric) throw new Error(`metric ${id} not found`);
  return metric;
};

const frontierNames = (points: EvaluationRow[], x: ParetoMetric, y: ParetoMetric): string[] =>
  buildParetoPoints(points, x, y)
    .filter((p) => p.onFrontier)
    .map((p) => p.name)
    .sort();

describe('deriveParetoMetrics', () => {
  it('always offers cost, latency and tokens (minimized) plus one option per evaluator (maximized)', () => {
    const points = [point({ name: 'a', evaluators: { reward: 1, safety: 1 } })];
    const metrics = deriveParetoMetrics(points);
    // Evaluator ids use the API vocabulary (`evaluators.<name>`) so they match the group's saved axes.
    expect(metrics.map((m) => m.id)).toEqual([
      'cost_usd',
      'latency_ms',
      'tokens',
      'evaluators.reward',
      'evaluators.safety',
    ]);
    expect(getMetric(metrics, 'cost_usd').direction).toBe('min');
    expect(getMetric(metrics, 'latency_ms').direction).toBe('min');
    expect(getMetric(metrics, 'tokens').direction).toBe('min');
    expect(getMetric(metrics, 'evaluators.reward').direction).toBe('max');
  });

  it('offers only cost, latency and tokens when no evaluators are present', () => {
    expect(deriveParetoMetrics([point({ name: 'a' })]).map((m) => m.id)).toEqual([
      'cost_usd',
      'latency_ms',
      'tokens',
    ]);
  });

  it('reads tokens from the mean token rollup', () => {
    const tokens = getMetric(deriveParetoMetrics([]), 'tokens');
    expect(tokens.label).toBe('Tokens');
    expect(tokens.accessor(point({ name: 'a', tokens: 1200 }))).toBe(1200);
  });
});

describe('buildParetoPoints', () => {
  it('marks the non-dominated set for two minimized axes (cost vs latency)', () => {
    const points = [
      point({ name: 'A', cost: 1, latency: 4 }), // cheapest -> frontier
      point({ name: 'B', cost: 2, latency: 2 }), // balanced -> frontier
      point({ name: 'C', cost: 4, latency: 1 }), // fastest -> frontier
      point({ name: 'D', cost: 3, latency: 3 }), // dominated by B
    ];
    const metrics = deriveParetoMetrics(points);
    expect(
      frontierNames(points, getMetric(metrics, 'cost_usd'), getMetric(metrics, 'latency_ms'))
    ).toEqual(['A', 'B', 'C']);
  });

  it('respects mixed directions: minimize cost, maximize an evaluator score', () => {
    const points = [
      point({ name: 'A', cost: 1, evaluators: { reward: 0.5 } }), // cheapest -> frontier
      point({ name: 'B', cost: 2, evaluators: { reward: 0.9 } }), // most accurate -> frontier
      point({ name: 'C', cost: 2, evaluators: { reward: 0.4 } }), // dominated by A
    ];
    const metrics = deriveParetoMetrics(points);
    expect(
      frontierNames(points, getMetric(metrics, 'cost_usd'), getMetric(metrics, 'evaluators.reward'))
    ).toEqual(['A', 'B']);
  });

  it('drops points missing either selected metric', () => {
    const points = [
      point({ name: 'A', cost: 1, latency: 1 }),
      point({ name: 'B', cost: 2 }), // no latency -> excluded
    ];
    const metrics = deriveParetoMetrics(points);
    const plotted = buildParetoPoints(
      points,
      getMetric(metrics, 'cost_usd'),
      getMetric(metrics, 'latency_ms')
    );
    expect(plotted.map((p) => p.name)).toEqual(['A']);
  });
});

describe('resolveParetoAxes', () => {
  const axisIds = (
    points: EvaluationRow[],
    preferredX = 'cost_usd',
    preferredY = 'latency_ms'
  ): [string, string] | undefined => {
    const axes = resolveParetoAxes(points, deriveParetoMetrics(points), preferredX, preferredY);
    return axes && [axes.x.id, axes.y.id];
  };

  it('keeps the preferred axes when they plot at least one evaluation', () => {
    const points = [
      point({ name: 'A', cost: 1, latency: 2, tokens: 100, evaluators: { reward: 0.5 } }),
      point({ name: 'B', latency: 3, tokens: 200, evaluators: { reward: 0.7 } }),
    ];
    expect(axisIds(points)).toEqual(['cost_usd', 'latency_ms']);
  });

  it('keeps a saved evaluator-vs-tokens selection', () => {
    const points = [
      point({ name: 'A', cost: 1, latency: 2, tokens: 100, evaluators: { reward: 1 } }),
    ];
    expect(axisIds(points, 'tokens', 'evaluators.reward')).toEqual(['tokens', 'evaluators.reward']);
  });

  it('falls back to an evaluator score vs tokens when there is no cost', () => {
    const points = [
      point({ name: 'A', latency: 0, tokens: 100, evaluators: { accuracy: 0.5, reward: 0.8 } }),
      point({ name: 'B', latency: 0, tokens: 300, evaluators: { accuracy: 0.9, reward: 0.6 } }),
    ];
    expect(axisIds(points)).toEqual(['evaluators.accuracy', 'tokens']);
  });

  it('prefers cost over tokens for the fallback when cost is recorded', () => {
    const points = [point({ name: 'A', cost: 0.01, tokens: 100, evaluators: { reward: 0.5 } })];
    expect(axisIds(points)).toEqual(['evaluators.reward', 'cost_usd']);
  });

  it('pairs two resource metrics when no evaluator has scores', () => {
    const points = [point({ name: 'A', latency: 5, tokens: 100 })];
    expect(axisIds(points)).toEqual(['latency_ms', 'tokens']);
  });

  it('keeps the preferred axes while there are no rows yet or nothing plottable', () => {
    expect(axisIds([])).toEqual(['cost_usd', 'latency_ms']);
    expect(axisIds([point({ name: 'A', tokens: 100 })])).toEqual(['cost_usd', 'latency_ms']);
  });

  it('falls back to the first metrics when a saved evaluator is not in the data', () => {
    const points = [point({ name: 'A', cost: 1, latency: 2 })];
    expect(axisIds(points, 'evaluators.gone', 'evaluators.also_gone')).toEqual([
      'cost_usd',
      'latency_ms',
    ]);
  });
});
