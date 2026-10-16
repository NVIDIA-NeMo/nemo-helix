// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EM_DASH } from '@nemo/common/src/constants';

/**
 * Hand-written TypeScript mirror of the Data Designer `DatasetProfilerResults`
 * Pydantic model (data_designer.config.analysis.dataset_profiler).
 *
 */

/** Sentinel emitted by the profiler when a statistic could not be computed. */
export const MISSING_VALUE = {
  CALCULATION_FAILED: '--',
  OUTPUT_FORMAT_ERROR: 'output_format_error',
} as const;

export type MissingValue = (typeof MISSING_VALUE)[keyof typeof MISSING_VALUE];

/** A numeric statistic that may instead be a {@link MissingValue} sentinel. */
export type MaybeMissing<T> = T | MissingValue;

export const isMissingValue = (value: unknown): value is MissingValue =>
  value === MISSING_VALUE.CALCULATION_FAILED || value === MISSING_VALUE.OUTPUT_FORMAT_ERROR;

export interface CategoricalHistogramData {
  categories: Array<string | number>;
  counts: number[];
}

export interface CategoricalDistribution {
  most_common_value: string | number;
  least_common_value: string | number;
  histogram: CategoricalHistogramData;
}

export interface NumericalDistribution {
  min: number;
  max: number;
  mean: number;
  stddev: number;
  median: number;
}

/** Fields shared by every column-statistics variant (GeneralColumnStatistics). */
interface BaseColumnStatistics {
  column_name: string;
  num_records: MaybeMissing<number>;
  num_null: MaybeMissing<number>;
  num_unique: MaybeMissing<number>;
  pyarrow_dtype: string;
  simple_dtype: string;
}

export interface GeneralColumnStatistics extends BaseColumnStatistics {
  column_type: 'general';
}

/** Token-usage metrics shared by all LLM-backed column types. */
interface LLMColumnStatisticsBase extends BaseColumnStatistics {
  output_tokens_mean: MaybeMissing<number>;
  output_tokens_median: MaybeMissing<number>;
  output_tokens_stddev: MaybeMissing<number>;
  input_tokens_mean: MaybeMissing<number>;
  input_tokens_median: MaybeMissing<number>;
  input_tokens_stddev: MaybeMissing<number>;
}

export interface LLMTextColumnStatistics extends LLMColumnStatisticsBase {
  column_type: 'llm-text';
}

export interface LLMCodeColumnStatistics extends LLMColumnStatisticsBase {
  column_type: 'llm-code';
}

export interface LLMStructuredColumnStatistics extends LLMColumnStatisticsBase {
  column_type: 'llm-structured';
}

export interface LLMJudgedColumnStatistics extends LLMColumnStatisticsBase {
  column_type: 'llm-judge';
}

export interface SamplerColumnStatistics extends BaseColumnStatistics {
  column_type: 'sampler';
  sampler_type: string;
  distribution_type: 'categorical' | 'numerical' | 'text' | 'other' | 'unknown';
  distribution: CategoricalDistribution | NumericalDistribution | MissingValue | null;
}

export interface SeedDatasetColumnStatistics extends BaseColumnStatistics {
  column_type: 'seed-dataset';
}

export interface ValidationColumnStatistics extends BaseColumnStatistics {
  column_type: 'validation';
  num_valid_records: MaybeMissing<number>;
}

export interface ExpressionColumnStatistics extends BaseColumnStatistics {
  column_type: 'expression';
}

/**
 * Plugin-provided column generators dynamically register General-shaped
 * statistics classes with their own `column_type`, so an unknown discriminator
 * still carries the base fields.
 */
export interface UnknownColumnStatistics extends BaseColumnStatistics {
  column_type: string;
}

export type ColumnStatistics =
  | GeneralColumnStatistics
  | LLMTextColumnStatistics
  | LLMCodeColumnStatistics
  | LLMStructuredColumnStatistics
  | LLMJudgedColumnStatistics
  | SamplerColumnStatistics
  | SeedDatasetColumnStatistics
  | ValidationColumnStatistics
  | ExpressionColumnStatistics
  | UnknownColumnStatistics;

export interface DatasetProfilerResults {
  num_records: number;
  target_num_records: number;
  column_statistics: ColumnStatistics[];
  side_effect_column_names?: string[] | null;
  // Advanced profiler results (e.g. JudgeScoreProfilerResults). Rendered as a
  // follow-up; typed loosely until the schema stabilizes.
  column_profiles?: unknown[] | null;
}

const LLM_COLUMN_TYPES = new Set(['llm-text', 'llm-code', 'llm-structured', 'llm-judge']);

export type LLMColumnStatistics =
  | LLMTextColumnStatistics
  | LLMCodeColumnStatistics
  | LLMStructuredColumnStatistics
  | LLMJudgedColumnStatistics;

export const isLLMColumnStatistics = (stats: ColumnStatistics): stats is LLMColumnStatistics =>
  LLM_COLUMN_TYPES.has(stats.column_type);

export const isValidationColumnStatistics = (
  stats: ColumnStatistics
): stats is ValidationColumnStatistics => stats.column_type === 'validation';

export const isSamplerColumnStatistics = (
  stats: ColumnStatistics
): stats is SamplerColumnStatistics => stats.column_type === 'sampler';

/** Completion percentage of the dataset (mirrors `percent_complete`). */
export const getPercentComplete = (results: DatasetProfilerResults): number => {
  if (results.target_num_records <= 0) {
    return 0;
  }
  const percent = (100 * results.num_records) / results.target_num_records;
  return Math.max(0, Math.min(100, percent));
};

/** `num_unique / num_records` as a percentage, or undefined when unavailable. */
export const getPercentUnique = (stats: ColumnStatistics): number | undefined => {
  if (
    isMissingValue(stats.num_unique) ||
    isMissingValue(stats.num_records) ||
    stats.num_records <= 0
  ) {
    return undefined;
  }
  return (100 * stats.num_unique) / stats.num_records;
};

/** `num_null / num_records` as a percentage, or undefined when unavailable. */
export const getPercentNull = (stats: ColumnStatistics): number | undefined => {
  if (
    isMissingValue(stats.num_null) ||
    isMissingValue(stats.num_records) ||
    stats.num_records <= 0
  ) {
    return undefined;
  }
  return (100 * stats.num_null) / stats.num_records;
};

/** Render a sentinel-or-undefined value as display text, or `null` if it's a real value. */
const formatSentinel = (value: MaybeMissing<number> | null | undefined): string | null => {
  if (value == null) {
    return EM_DASH;
  }
  if (value === MISSING_VALUE.OUTPUT_FORMAT_ERROR) {
    return 'format error';
  }
  if (value === MISSING_VALUE.CALCULATION_FAILED) {
    return EM_DASH;
  }
  return null;
};

/** Format a whole-number statistic, surfacing missing-value sentinels as text. */
export const formatStatCount = (value: MaybeMissing<number> | null | undefined): string =>
  formatSentinel(value) ?? (value as number).toLocaleString();

/** Format a (possibly fractional) statistic, surfacing missing-value sentinels as text. */
export const formatStatDecimal = (value: MaybeMissing<number> | null | undefined): string =>
  formatSentinel(value) ??
  (value as number).toLocaleString(undefined, { maximumFractionDigits: 1 });

/** Format a percentage produced by the getPercent* helpers. */
export const formatPercent = (value: number | undefined): string =>
  value == null ? EM_DASH : `${value.toFixed(1)}%`;

/** Human-readable label for a column's generator type (e.g. `sampler · category`). */
export const getColumnTypeLabel = (stats: ColumnStatistics): string =>
  isSamplerColumnStatistics(stats) ? `sampler · ${stats.sampler_type}` : stats.column_type;

export const isCategoricalDistribution = (
  distribution: CategoricalDistribution | NumericalDistribution
): distribution is CategoricalDistribution => 'histogram' in distribution;

/**
 * Categorical histogram for a sampler column, or `null` when the column isn't a
 * sampler, has no distribution, or its distribution isn't categorical. Use this
 * to decide whether a column should render a bar chart of its value counts.
 */
export const getCategoricalHistogram = (
  stats: ColumnStatistics
): CategoricalHistogramData | null => {
  if (!isSamplerColumnStatistics(stats)) {
    return null;
  }
  const { distribution } = stats;
  if (distribution == null || isMissingValue(distribution)) {
    return null;
  }
  return isCategoricalDistribution(distribution) ? distribution.histogram : null;
};

/**
 * Numerical distribution (min/max/mean/median/stddev) for a sampler column, or
 * `null` when the column isn't a sampler with a numerical distribution.
 */
export const getNumericalDistribution = (stats: ColumnStatistics): NumericalDistribution | null => {
  if (!isSamplerColumnStatistics(stats)) {
    return null;
  }
  const { distribution } = stats;
  if (distribution == null || isMissingValue(distribution)) {
    return null;
  }
  return isCategoricalDistribution(distribution) ? null : distribution;
};

const describeSamplerDistribution = (stats: SamplerColumnStatistics): string => {
  const { distribution } = stats;
  if (distribution == null) {
    return EM_DASH;
  }
  if (isMissingValue(distribution)) {
    return formatSentinel(distribution) ?? EM_DASH;
  }
  if (isCategoricalDistribution(distribution)) {
    return `Most common: ${distribution.most_common_value}`;
  }
  return `min ${formatStatDecimal(distribution.min)} · max ${formatStatDecimal(
    distribution.max
  )} · mean ${formatStatDecimal(distribution.mean)}`;
};

/**
 * Variant-specific one-line summary for a column's "Details" cell. LLM columns
 * surface token usage, validation columns surface valid-record counts, sampler
 * columns surface their distribution, and everything else falls back to a dash.
 */
export const describeColumnStats = (stats: ColumnStatistics): string => {
  if (isLLMColumnStatistics(stats)) {
    return `Tokens in/out (avg): ${formatStatDecimal(
      stats.input_tokens_mean
    )} / ${formatStatDecimal(stats.output_tokens_mean)}`;
  }
  if (isValidationColumnStatistics(stats)) {
    return `Valid records: ${formatStatCount(stats.num_valid_records)}`;
  }
  if (isSamplerColumnStatistics(stats)) {
    return describeSamplerDistribution(stats);
  }
  return EM_DASH;
};
