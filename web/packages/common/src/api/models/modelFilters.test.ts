// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  countActiveModelFilters,
  hasActiveModelFilters,
  MODEL_CAPABILITY_OPTIONS,
  MODEL_INPUT_MODALITY_OPTIONS,
  modelSizeFilter,
  toModelCapabilities,
  toModelModalities,
  toModelSizeBuckets,
  withSelectedOptions,
} from '@nemo/common/src/api/models/modelFilters';
import { ModelCapability } from '@nemo/sdk/generated/platform/schema/ModelCapability';
import { ModelModality } from '@nemo/sdk/generated/platform/schema/ModelModality';

const BILLION = 1_000_000_000;

describe('modelSizeFilter', () => {
  it.each([
    ['small', { $gte: 0, $lt: 3 * BILLION }],
    ['medium', { $gte: 3 * BILLION, $lt: 10 * BILLION }],
    ['large', { $gte: 10 * BILLION, $lt: 50 * BILLION }],
    ['xlarge', { $gte: 50 * BILLION }],
  ] as const)('bounds the %s bucket on its own', (bucket, expected) => {
    expect(modelSizeFilter([bucket])).toEqual(expected);
  });

  it('sends several buckets as a JSON-encoded $or in bucket order', () => {
    const filter = modelSizeFilter(['xlarge', 'small']);

    expect(filter).toEqual({
      $or: JSON.stringify([{ $gte: 0, $lt: 3 * BILLION }, { $gte: 50 * BILLION }]),
    });
  });

  it('sends nothing with no buckets', () => {
    expect(modelSizeFilter([])).toBeUndefined();
  });
});

describe('known value guards', () => {
  it('keeps known values and drops the rest', () => {
    expect(toModelSizeBuckets(['large', 'huge', 'small'])).toEqual(['large', 'small']);
    expect(toModelCapabilities(['reasoning', 'flying'])).toEqual(['reasoning']);
    expect(toModelModalities(['image', ''])).toEqual(['image']);
  });
});

describe('capability and input options', () => {
  it('offers every capability and modality the API accepts', () => {
    expect(MODEL_CAPABILITY_OPTIONS.map((option) => option.value)).toEqual(
      Object.values(ModelCapability)
    );
    expect(MODEL_INPUT_MODALITY_OPTIONS.map((option) => option.value)).toEqual(
      Object.values(ModelModality)
    );
  });

  it('labels values for people rather than the API', () => {
    expect(MODEL_CAPABILITY_OPTIONS).toContainEqual({
      value: 'tool_calling',
      label: 'Tool calling',
    });
    expect(MODEL_INPUT_MODALITY_OPTIONS).toContainEqual({ value: 'pdf', label: 'PDF' });
  });
});

describe('withSelectedOptions', () => {
  const options = [{ value: 'llama', label: 'llama' }];

  it('keeps the options when every selection is already listed', () => {
    expect(withSelectedOptions(options, ['llama'])).toBe(options);
  });

  it('prepends selections the loaded options do not include', () => {
    expect(withSelectedOptions(options, ['llama', 'gemma'])).toEqual([
      { value: 'gemma', label: 'gemma' },
      ...options,
    ]);
  });

  it('leaves the options alone with no selection', () => {
    expect(withSelectedOptions(options)).toBe(options);
  });
});

describe('active filter helpers', () => {
  it('counts the filters with at least one value', () => {
    expect(countActiveModelFilters({})).toBe(0);
    expect(countActiveModelFilters({ providers: [], sizes: ['small'] })).toBe(1);
    expect(
      countActiveModelFilters({
        providers: ['ws/p'],
        inputModalities: ['image', 'video'],
        sizes: ['small'],
      })
    ).toBe(3);
    expect(hasActiveModelFilters({ capabilities: ['reasoning'] })).toBe(true);
    expect(hasActiveModelFilters({ capabilities: [] })).toBe(false);
  });
});
