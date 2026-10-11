// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatEvaluatorScore } from '@nemo/common/src/utils/formatters';
import { evalDurationMs } from '@studio/api/evaluation/utils';
import {
  type MetricDirection,
  type ParetoMetric,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';

export const DURATION_METRIC: ParetoMetric = {
  id: 'eval_duration',
  label: 'Duration',
  direction: 'min',
  accessor: (row) => evalDurationMs(row.metadata),
};

const createdAtMs = (row: EvaluationRow): number => {
  const ms = row.created_at ? Date.parse(row.created_at) : NaN;
  return Number.isNaN(ms) ? Infinity : ms;
};

export const defaultBaselineName = (rows: readonly EvaluationRow[]): string | undefined =>
  rows.reduce<EvaluationRow | undefined>(
    (oldest, row) => (!oldest || createdAtMs(row) < createdAtMs(oldest) ? row : oldest),
    undefined
  )?.name;

const finiteValue = (metric: ParetoMetric, row: EvaluationRow): number | undefined => {
  const value = metric.accessor(row);
  return value != null && Number.isFinite(value) ? value : undefined;
};

/** `row`'s score minus the baseline's, or undefined for the baseline itself or a missing score. */
export const scoreDelta = (
  metric: ParetoMetric,
  row: EvaluationRow,
  baseline: EvaluationRow | undefined
): number | undefined => {
  if (!baseline || row.name === baseline.name) return undefined;
  const value = finiteValue(metric, row);
  const baselineValue = finiteValue(metric, baseline);
  return value === undefined || baselineValue === undefined ? undefined : value - baselineValue;
};

export type DeltaTone = 'better' | 'worse' | 'neutral';

export const describeDelta = (
  delta: number,
  direction: MetricDirection
): { text: string; tone: DeltaTone } => {
  const formatted = formatEvaluatorScore(delta);
  if (Number(formatted) === 0) return { text: '0', tone: 'neutral' };
  const improved = direction === 'max' ? delta > 0 : delta < 0;
  return { text: delta > 0 ? `+${formatted}` : formatted, tone: improved ? 'better' : 'worse' };
};

export interface ColumnExtremes {
  readonly best: number;
  readonly worst: number;
}

/** The column's best and worst values, or undefined when fewer than two distinct values leave nothing to pick. */
export const columnExtremes = (
  rows: readonly EvaluationRow[],
  metric: ParetoMetric
): ColumnExtremes | undefined => {
  const values = rows
    .map((row) => finiteValue(metric, row))
    .filter((value): value is number => value !== undefined);
  if (new Set(values).size < 2) return undefined;
  const [high, low] = [Math.max(...values), Math.min(...values)];
  return metric.direction === 'max' ? { best: high, worst: low } : { best: low, worst: high };
};
