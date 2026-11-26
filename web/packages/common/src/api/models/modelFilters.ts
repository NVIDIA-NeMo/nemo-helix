// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NumberFilter } from '@nemo/sdk/generated/platform/schema';
import { ModelCapability } from '@nemo/sdk/generated/platform/schema/ModelCapability';
import { ModelModality } from '@nemo/sdk/generated/platform/schema/ModelModality';

export type ModelSizeBucket = 'small' | 'medium' | 'large' | 'xlarge';

export interface ModelFilterValues {
  provider?: string;
  capability?: ModelCapability;
  inputModality?: ModelModality;
  size?: ModelSizeBucket;
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

export const toModelSizeBucket = (value?: string): ModelSizeBucket | undefined =>
  MODEL_SIZE_BUCKETS.find((bucket) => bucket.value === value)?.value;

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

export const toModelCapability = (value?: string): ModelCapability | undefined =>
  Object.values(ModelCapability).find((capability) => capability === value);

export const toModelModality = (value?: string): ModelModality | undefined =>
  Object.values(ModelModality).find((modality) => modality === value);

export const EMPTY_MODEL_FILTERS: ModelFilterValues = {};

export const countActiveModelFilters = (filters: ModelFilterValues): number =>
  [filters.provider, filters.capability, filters.inputModality, filters.size].filter(Boolean)
    .length;

export const hasActiveModelFilters = (filters: ModelFilterValues): boolean =>
  countActiveModelFilters(filters) > 0;

export const modelSizeFilter = (size: ModelSizeBucket): NumberFilter | undefined => {
  const bucket = MODEL_SIZE_BUCKETS.find((candidate) => candidate.value === size);
  if (!bucket) {
    return undefined;
  }
  if (bucket.maxParameters === undefined) {
    return { $gte: bucket.minParameters };
  }
  return { $gte: bucket.minParameters, $lt: bucket.maxParameters };
};

export const withSelectedOption = (
  options: ModelFilterOption[],
  selected?: string
): ModelFilterOption[] => {
  if (!selected || options.some((option) => option.value === selected)) return options;
  return [{ value: selected, label: selected }, ...options];
};
