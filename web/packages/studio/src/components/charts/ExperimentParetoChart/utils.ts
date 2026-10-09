// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

/** Which direction on an axis counts as "better": cost/latency/tokens minimize, evaluator scores maximize. */
export type MetricDirection = 'min' | 'max';

export interface ParetoMetric {
  /** Stable id in the API's metric vocabulary: `cost_usd`, `latency_ms`, `tokens`, or `evaluators.<name>`. */
  readonly id: string;
  readonly label: string;
  readonly direction: MetricDirection;
  readonly accessor: (row: EvaluationRow) => number | null | undefined;
}

const capitalize = (value: string): string =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : value;

/** Display label for a metric id, resolved from the id alone: `cost_usd` -> "Cost (USD)",
 * `latency_ms` -> "Latency (ms)", `tokens` -> "Tokens", `evaluators.<name>` -> the capitalized name. */
export function metricLabel(id: string): string {
  if (id === 'cost_usd') return 'Cost (USD)';
  if (id === 'latency_ms') return 'Latency (ms)';
  if (id === 'tokens') return 'Tokens';
  return capitalize(id.startsWith('evaluators.') ? id.slice('evaluators.'.length) : id);
}

const COST_METRIC: ParetoMetric = {
  id: 'cost_usd',
  label: metricLabel('cost_usd'),
  direction: 'min',
  accessor: (row) => row.cost_usd?.mean,
};

const LATENCY_METRIC: ParetoMetric = {
  id: 'latency_ms',
  label: metricLabel('latency_ms'),
  direction: 'min',
  accessor: (row) => row.latency_ms?.mean,
};

const TOKENS_METRIC: ParetoMetric = {
  id: 'tokens',
  label: metricLabel('tokens'),
  direction: 'min',
  accessor: (row) => row.tokens?.mean,
};

/** Metrics selectable on either axis: cost, latency and tokens (minimized) plus one per evaluator seen
 * in the data (maximized). Evaluator names are dynamic, so they're derived from the rows. */
export function deriveParetoMetrics(rows: readonly EvaluationRow[]): ParetoMetric[] {
  const evaluatorNames = [
    ...new Set(rows.flatMap((row) => Object.keys(row.aggregate_scores ?? {}))),
  ].sort();
  const evaluatorMetrics = evaluatorNames.map<ParetoMetric>((name) => ({
    id: `evaluators.${name}`,
    label: metricLabel(`evaluators.${name}`),
    direction: 'max',
    accessor: (row) => row.aggregate_scores?.[name]?.mean,
  }));
  return [COST_METRIC, LATENCY_METRIC, TOKENS_METRIC, ...evaluatorMetrics];
}

export interface ParetoPlotPoint {
  readonly name: string;
  readonly x: number;
  readonly y: number;
  /** True when no other evaluation dominates this one on both axes. */
  readonly onFrontier: boolean;
}

interface Coords {
  readonly x: number;
  readonly y: number;
}

/** Whether `b` dominates `a`: at least as good on both axes and strictly better on at least one. */
function dominates(a: Coords, b: Coords, xDir: MetricDirection, yDir: MetricDirection): boolean {
  const atLeastAsGood = (av: number, bv: number, dir: MetricDirection): boolean =>
    dir === 'min' ? bv <= av : bv >= av;
  const strictlyBetter = (av: number, bv: number, dir: MetricDirection): boolean =>
    dir === 'min' ? bv < av : bv > av;
  return (
    atLeastAsGood(a.x, b.x, xDir) &&
    atLeastAsGood(a.y, b.y, yDir) &&
    (strictlyBetter(a.x, b.x, xDir) || strictlyBetter(a.y, b.y, yDir))
  );
}

/**
 * Build plot points for two metrics and flag which lie on the Pareto frontier — the evaluations not
 * dominated by any other on both axes. Points missing either metric (non-finite) are dropped.
 */
export function buildParetoPoints(
  rows: readonly EvaluationRow[],
  xMetric: ParetoMetric,
  yMetric: ParetoMetric
): ParetoPlotPoint[] {
  const coords = rows
    .map((row): { name: string; x: number; y: number } | null => {
      const x = xMetric.accessor(row);
      const y = yMetric.accessor(row);
      if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) return null;
      return { name: row.name, x, y };
    })
    .filter((point): point is { name: string; x: number; y: number } => point !== null);

  return coords.map((point) => ({
    ...point,
    onFrontier: !coords.some(
      (other) => other !== point && dominates(point, other, xMetric.direction, yMetric.direction)
    ),
  }));
}

export interface ParetoAxes {
  readonly x: ParetoMetric;
  readonly y: ParetoMetric;
}

const hasPlottableValue = (rows: readonly EvaluationRow[], metric: ParetoMetric): boolean =>
  rows.some((row) => Number.isFinite(metric.accessor(row)));

/** Resource metrics in the order they're tried for the fallback Y axis. Latency comes last because
 * agent evals record it as 0 today, which plots but separates nothing. */
const FALLBACK_Y_METRICS: readonly ParetoMetric[] = [COST_METRIC, TOKENS_METRIC, LATENCY_METRIC];

/** The preferred (saved) axes when they plot at least one evaluation, otherwise the first evaluator
 * score against the first resource metric that has values. */
export function resolveParetoAxes(
  rows: readonly EvaluationRow[],
  metrics: readonly ParetoMetric[],
  preferredX: string,
  preferredY: string
): ParetoAxes | undefined {
  const x = metrics.find((m) => m.id === preferredX) ?? metrics[0];
  const y = metrics.find((m) => m.id === preferredY) ?? metrics[1] ?? metrics[0];
  if (!x || !y) return undefined;
  if (rows.length === 0 || buildParetoPoints(rows, x, y).length > 0) return { x, y };

  const plottable = metrics.filter((m) => hasPlottableValue(rows, m));
  const fallbackX = plottable.find((m) => m.direction === 'max') ?? plottable[0];
  const fallbackY =
    FALLBACK_Y_METRICS.find((m) => m !== fallbackX && plottable.includes(m)) ??
    plottable.find((m) => m !== fallbackX);
  return fallbackX && fallbackY ? { x: fallbackX, y: fallbackY } : { x, y };
}
