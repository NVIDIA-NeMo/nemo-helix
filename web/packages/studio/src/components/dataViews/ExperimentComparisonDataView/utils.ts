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
  if (new Set(values).size < 2) {
    return undefined;
  }
  const high = Math.max(...values);
  const low = Math.min(...values);
  if (metric.direction === 'max') {
    return { best: high, worst: low };
  }
  return { best: low, worst: high };
};

const ACRONYMS: Readonly<Record<string, string>> = { llm: 'LLM' };

const humanize = (value: string): string => {
  const words = value.split(/[-_\s]+/).filter(Boolean);
  const text = words.map((word) => ACRONYMS[word.toLowerCase()] ?? word.toLowerCase()).join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
};

/** Readable header for an evaluator score: `llm-judge.behaviour` -> "Behaviour (LLM judge)", `string-check.string-check` -> "String check". */
export const scoreHeader = (metricId: string): string => {
  const key = metricId.replace(/^evaluators\./, '');
  const dot = key.indexOf('.');
  if (dot === -1) {
    return humanize(key);
  }
  const metricType = key.slice(0, dot);
  const output = key.slice(dot + 1);
  if (output === metricType) {
    return humanize(output);
  }
  return `${humanize(output)} (${humanize(metricType)})`;
};
