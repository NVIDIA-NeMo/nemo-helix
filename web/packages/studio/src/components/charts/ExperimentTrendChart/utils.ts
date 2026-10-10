// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDurationMs } from '@nemo/common/src/utils/date';
import { snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import { evalDurationMs } from '@studio/api/evaluation/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { evaluatorLabel } from '@studio/routes/agents/AgentDetailRoute/evaluations/formatRollups';

export interface TrendMetric {
  /** Stable id in the API's metric vocabulary: `cost_usd`, `latency_ms`, `tokens`, or
   * `evaluators.<name>` — the same vocabulary the Pareto chart's axes are named in. */
  readonly id: string;
  readonly label: string;
  /** Formats a value in the metric's own units, for the tooltip. */
  readonly format: (value: number) => string;
  readonly accessor: (row: EvaluationRow) => number | null | undefined;
}

/** Compact big tick values so they don't collide with the axis title (16000 -> "16K"); keep small
 * values precise so close cost/score ticks stay distinct. */
export const formatAxisTick = (value: number): string =>
  Math.abs(value) >= 1000
    ? value.toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: 1 })
    : value.toLocaleString(undefined, { maximumFractionDigits: 3 });

const formatCost = (value: number): string =>
  `$${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}`;

const formatLatency = (value: number): string => `${Math.round(value).toLocaleString()} ms`;

const formatTokens = (value: number): string =>
  value.toLocaleString(undefined, { maximumFractionDigits: 0 });

const formatScore = (value: number): string =>
  value.toLocaleString(undefined, { maximumFractionDigits: 3 });

/**
 * The evaluation-level measures graphable alongside evaluator scores.
 *
 * Cost, latency and tokens are typed rollups the API aggregates onto an evaluation. Duration is not
 * one: the evaluator stamps the run's wall-clock seconds into `metadata.eval_duration_sec` at
 * publish time, which is why `evalDurationMs` reads it rather than a field. It is therefore present
 * only for runs that published successfully, and `buildTrendPoints` drops the rest.
 */
const RESOURCE_METRICS: readonly TrendMetric[] = [
  {
    id: 'duration_ms',
    label: 'Duration',
    format: (value) => formatDurationMs(value),
    accessor: (row) => evalDurationMs(row.metadata),
  },
  {
    id: 'cost_usd',
    label: 'Cost (USD)',
    format: formatCost,
    accessor: (row) => row.cost_usd?.mean,
  },
  {
    id: 'latency_ms',
    label: 'Latency (ms)',
    format: formatLatency,
    accessor: (row) => row.latency_ms?.mean,
  },
  { id: 'tokens', label: 'Tokens', format: formatTokens, accessor: (row) => row.tokens?.mean },
];

/**
 * Metrics selectable on the trend: one per evaluator seen in the data (alphabetized) followed by
 * the cost, latency and token rollups. Evaluator names are dynamic, so they are derived from the
 * rows — the same shape as `deriveParetoMetrics`. Scorers lead because they are what an experiment
 * is usually judged on.
 */
export function deriveTrendMetrics(rows: readonly EvaluationRow[]): TrendMetric[] {
  const evaluatorNames = [
    ...new Set(rows.flatMap((row) => Object.keys(row.aggregate_scores ?? {}))),
  ];
  const shortLabels = evaluatorNames.map((name) => snakeCaseToTitleCase(evaluatorLabel(name)));
  // `correctness.score` and `tool_use.score` both shorten to "Score"; keep the full key when they clash.
  const isAmbiguous = (label: string) => shortLabels.filter((l) => l === label).length > 1;
  const evaluatorMetrics = evaluatorNames
    .map<TrendMetric>((name, index) => ({
      id: `evaluators.${name}`,
      label: isAmbiguous(shortLabels[index] ?? '')
        ? snakeCaseToTitleCase(name.replaceAll('.', '_'))
        : (shortLabels[index] ?? name),
      // Scores carry no scale metadata, so they are shown as-is rather than as a percentage.
      format: formatScore,
      accessor: (row) => row.aggregate_scores?.[name]?.mean,
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
  return [...evaluatorMetrics, ...RESOURCE_METRICS];
}

export interface TrendPlotPoint {
  readonly name: string;
  /** Epoch millis of the evaluation's `created_at`. The x-axis is a real time scale, so runs that
   * cluster into one afternoon sit together instead of being spread evenly across the chart. */
  readonly x: number;
  readonly y: number;
}

/**
 * Build the plotted series for one metric, oldest first.
 *
 * A row is dropped when it has no parseable `created_at` — nothing to place it at on the axis — or
 * no finite value for the metric, mirroring how `buildParetoPoints` drops points missing an axis.
 */
export function buildTrendPoints(
  rows: readonly EvaluationRow[],
  metric: TrendMetric
): TrendPlotPoint[] {
  return rows
    .map((row): TrendPlotPoint | null => {
      const x = Date.parse(row.created_at ?? '');
      const y = metric.accessor(row);
      if (!Number.isFinite(x) || y == null || !Number.isFinite(y)) return null;
      return { name: row.name, x, y };
    })
    .filter((point): point is TrendPlotPoint => point !== null)
    .sort((a, b) => a.x - b.x);
}

/** Axis label for a point in time. Date alone: a trend spanning weeks needs no clock time, and the
 * tooltip carries the fuller timestamp for runs that land on the same day. */
export const formatTimeTick = (value: number): string =>
  new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

export const formatTimestamp = (value: number): string =>
  new Date(value).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
