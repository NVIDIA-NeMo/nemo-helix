// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  countActiveModelFilters,
  hasActiveModelFilters,
  MODEL_CAPABILITY_OPTIONS,
  MODEL_INPUT_MODALITY_OPTIONS,
  modelSizeFilter,
  toModelCapability,
  toModelModality,
  toModelSizeBucket,
  withSelectedOption,
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
  ] as const)('bounds the %s bucket', (bucket, expected) => {
    expect(modelSizeFilter(bucket)).toEqual(expected);
  });
});

describe('toModelSizeBucket', () => {
  it('passes a known bucket through', () => {
    expect(toModelSizeBucket('large')).toBe('large');
  });

  it.each([undefined, '', 'huge'])('drops %j', (value) => {
    expect(toModelSizeBucket(value)).toBeUndefined();
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

  it('passes known values through and drops the rest', () => {
    expect(toModelCapability('reasoning')).toBe('reasoning');
    expect(toModelCapability('flying')).toBeUndefined();
    expect(toModelModality('image')).toBe('image');
    expect(toModelModality(undefined)).toBeUndefined();
  });
});

describe('withSelectedOption', () => {
  const options = [{ value: 'llama', label: 'llama' }];

  it('keeps the options when the selection is already listed', () => {
    expect(withSelectedOption(options, 'llama')).toBe(options);
  });

  it('prepends a selection the loaded options do not include', () => {
    expect(withSelectedOption(options, 'gemma')).toEqual([
      { value: 'gemma', label: 'gemma' },
      ...options,
    ]);
  });

  it('leaves the options alone with no selection', () => {
    expect(withSelectedOption(options)).toBe(options);
  });
});

describe('active filter helpers', () => {
  it('counts only the filters that are set', () => {
    expect(countActiveModelFilters({})).toBe(0);
    expect(
      countActiveModelFilters({ provider: 'ws/p', inputModality: 'image', size: 'small' })
    ).toBe(3);
    expect(hasActiveModelFilters({ capability: 'reasoning' })).toBe(true);
    expect(hasActiveModelFilters({})).toBe(false);
  });
});
