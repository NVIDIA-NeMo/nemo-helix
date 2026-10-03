// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

export type ModelSizeBucket = 'small' | 'medium' | 'large' | 'xlarge';

export interface ModelFilterValues {
  provider?: string;
  family?: string;
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
  familyOptions: ModelFilterOption[];
  sizeOptions: ModelFilterOption[];
  providersLoading: boolean;
}

interface ModelSizeBucketDefinition {
  value: ModelSizeBucket;
  label: string;
  minParameters: number;
  maxParameters: number;
}

const BILLION = 1_000_000_000;

export const MODEL_SIZE_BUCKETS: readonly ModelSizeBucketDefinition[] = [
  { value: 'small', label: 'Up to 3B', minParameters: 0, maxParameters: 3 * BILLION },
  { value: 'medium', label: '3B to 10B', minParameters: 3 * BILLION, maxParameters: 10 * BILLION },
  { value: 'large', label: '10B to 50B', minParameters: 10 * BILLION, maxParameters: 50 * BILLION },
  { value: 'xlarge', label: 'Over 50B', minParameters: 50 * BILLION, maxParameters: Infinity },
];

export const MODEL_SIZE_OPTIONS: ModelFilterOption[] = MODEL_SIZE_BUCKETS.map(
  ({ value, label }) => ({ value, label })
);

export const toModelSizeBucket = (value?: string): ModelSizeBucket | undefined =>
  MODEL_SIZE_BUCKETS.find((bucket) => bucket.value === value)?.value;

export const EMPTY_MODEL_FILTERS: ModelFilterValues = {};

export const hasActiveModelFilters = (filters: ModelFilterValues): boolean =>
  Boolean(filters.provider || filters.family || filters.size);

export const countActiveModelFilters = (filters: ModelFilterValues): number =>
  [filters.provider, filters.family, filters.size].filter(Boolean).length;

/** A model with no recorded size never matches a size filter, so it cannot be mistaken for a small one. */
export const matchesModelSize = (model: ModelEntity, size: ModelSizeBucket): boolean => {
  const parameters = model.spec?.base_num_parameters;
  if (typeof parameters !== 'number') return false;
  const bucket = MODEL_SIZE_BUCKETS.find((candidate) => candidate.value === size);
  if (!bucket) return false;
  return parameters >= bucket.minParameters && parameters < bucket.maxParameters;
};

export const collectModelFamilies = (models: readonly ModelEntity[]): ModelFilterOption[] => {
  const families = new Set<string>();
  models.forEach((model) => {
    const family = model.spec?.family;
    if (family) families.add(family);
  });
  return [...families]
    .sort((a, b) => a.localeCompare(b))
    .map((family) => ({ value: family, label: family }));
};

export const withSelectedOption = (
  options: ModelFilterOption[],
  selected?: string
): ModelFilterOption[] => {
  if (!selected || options.some((option) => option.value === selected)) return options;
  return [{ value: selected, label: selected }, ...options];
};
