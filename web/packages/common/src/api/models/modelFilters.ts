// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NumberFilter } from '@nemo/sdk/generated/platform/schema';
import { ModelCapability } from '@nemo/sdk/generated/platform/schema/ModelCapability';
import { ModelModality } from '@nemo/sdk/generated/platform/schema/ModelModality';

export type ModelSizeBucket = 'small' | 'medium' | 'large' | 'xlarge';

/** Each list matches models with any of its values; the lists combine with AND. */
export interface ModelFilterValues {
  providers?: string[];
  capabilities?: ModelCapability[];
  inputModalities?: ModelModality[];
  sizes?: ModelSizeBucket[];
}

export interface ModelFilterOption {
  value: string;
  label: string;
}

export interface ModelFilterControls {
  values: ModelFilterValues;
  onChange: (values: ModelFilterValues) => void;
  providerOptions: ModelFilterOption[];
  capabilityOptions: ModelFilterOption[];
  inputModalityOptions: ModelFilterOption[];
  sizeOptions: ModelFilterOption[];
  providersLoading: boolean;
}

/** One `parameter_count` range, or several sent as a JSON-encoded `$or`. */
export type ParameterCountFilter = NumberFilter | { $or: string };

interface ModelSizeBucketDefinition {
  value: ModelSizeBucket;
  label: string;
  minParameters: number;
  maxParameters?: number;
}

const BILLION = 1_000_000_000;

export const MODEL_SIZE_BUCKETS: readonly ModelSizeBucketDefinition[] = [
  { value: 'small', label: 'Up to 3B', minParameters: 0, maxParameters: 3 * BILLION },
  { value: 'medium', label: '3B to 10B', minParameters: 3 * BILLION, maxParameters: 10 * BILLION },
  { value: 'large', label: '10B to 50B', minParameters: 10 * BILLION, maxParameters: 50 * BILLION },
  { value: 'xlarge', label: 'Over 50B', minParameters: 50 * BILLION },
];

export const MODEL_SIZE_OPTIONS: ModelFilterOption[] = MODEL_SIZE_BUCKETS.map(
  ({ value, label }) => ({ value, label })
);

const CAPABILITY_LABELS: Record<ModelCapability, string> = {
  tool_calling: 'Tool calling',
  reasoning: 'Reasoning',
  structured_output: 'Structured output',
  embedding: 'Embedding',
  reranking: 'Reranking',
};

const MODALITY_LABELS: Record<ModelModality, string> = {
  text: 'Text',
  image: 'Image',
  audio: 'Audio',
  video: 'Video',
  pdf: 'PDF',
};

export const MODEL_CAPABILITY_OPTIONS: ModelFilterOption[] = Object.values(ModelCapability).map(
  (value) => ({ value, label: CAPABILITY_LABELS[value] })
);

export const MODEL_INPUT_MODALITY_OPTIONS: ModelFilterOption[] = Object.values(ModelModality).map(
  (value) => ({ value, label: MODALITY_LABELS[value] })
);

const knownValues = <T extends string>(allowed: readonly T[], values: readonly string[]): T[] =>
  values.filter((value): value is T => (allowed as readonly string[]).includes(value));

export const toModelSizeBuckets = (values: readonly string[]): ModelSizeBucket[] =>
  knownValues(
    MODEL_SIZE_BUCKETS.map((bucket) => bucket.value),
    values
  );

export const toModelCapabilities = (values: readonly string[]): ModelCapability[] =>
  knownValues(Object.values(ModelCapability), values);

export const toModelModalities = (values: readonly string[]): ModelModality[] =>
  knownValues(Object.values(ModelModality), values);

export const EMPTY_MODEL_FILTERS: ModelFilterValues = {};

export const countActiveModelFilters = (filters: ModelFilterValues): number =>
  [filters.providers, filters.capabilities, filters.inputModalities, filters.sizes].filter(
    (values) => !!values?.length
  ).length;

export const hasActiveModelFilters = (filters: ModelFilterValues): boolean =>
  countActiveModelFilters(filters) > 0;

const bucketRange = (bucket: ModelSizeBucketDefinition): NumberFilter => {
  if (bucket.maxParameters === undefined) {
    return { $gte: bucket.minParameters };
  }
  return { $gte: bucket.minParameters, $lt: bucket.maxParameters };
};

export const modelSizeFilter = (
  sizes: readonly ModelSizeBucket[]
): ParameterCountFilter | undefined => {
  const ranges = MODEL_SIZE_BUCKETS.filter((bucket) => sizes.includes(bucket.value)).map(
    bucketRange
  );
  if (ranges.length === 0) {
    return undefined;
  }
  if (ranges.length === 1) {
    return ranges[0];
  }
  // qs cannot encode an array of objects in a query string; the API decodes JSON-valued filters.
  return { $or: JSON.stringify(ranges) };
};

export const withSelectedOptions = (
  options: ModelFilterOption[],
  selected: readonly string[] = []
): ModelFilterOption[] => {
  const missing = selected.filter((value) => !options.some((option) => option.value === value));
  if (missing.length === 0) {
    return options;
  }
  return [...missing.map((value) => ({ value, label: value })), ...options];
};
