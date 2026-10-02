// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  collectModelFamilies,
  countActiveModelFilters,
  hasActiveModelFilters,
  matchesModelSize,
  withSelectedOption,
} from '@nemo/common/src/api/models/modelFilters';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

const BILLION = 1_000_000_000;

const makeModel = (spec?: { family?: string; base_num_parameters?: number }): ModelEntity =>
  ({ id: 'm', name: 'm', workspace: 'ws', spec }) as ModelEntity;

describe('matchesModelSize', () => {
  it.each([
    [1 * BILLION, 'small'],
    [3 * BILLION, 'medium'],
    [9 * BILLION, 'medium'],
    [10 * BILLION, 'large'],
    [49 * BILLION, 'large'],
    [50 * BILLION, 'xlarge'],
    [400 * BILLION, 'xlarge'],
  ] as const)('puts %i parameters in the %s bucket only', (parameters, bucket) => {
    const model = makeModel({ base_num_parameters: parameters });

    const matching = (['small', 'medium', 'large', 'xlarge'] as const).filter((candidate) =>
      matchesModelSize(model, candidate)
    );

    expect(matching).toEqual([bucket]);
  });

  it('never matches a model with no recorded size', () => {
    expect(matchesModelSize(makeModel(), 'small')).toBe(false);
    expect(matchesModelSize(makeModel({ family: 'llama' }), 'xlarge')).toBe(false);
  });
});

describe('collectModelFamilies', () => {
  it('returns each family once, sorted', () => {
    const models = [
      makeModel({ family: 'mixtral' }),
      makeModel({ family: 'llama' }),
      makeModel({ family: 'mixtral' }),
    ];

    expect(collectModelFamilies(models)).toEqual([
      { value: 'llama', label: 'llama' },
      { value: 'mixtral', label: 'mixtral' },
    ]);
  });

  it('skips models without a family', () => {
    expect(
      collectModelFamilies([makeModel(), makeModel({ base_num_parameters: BILLION })])
    ).toEqual([]);
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
    expect(countActiveModelFilters({ provider: 'ws/p', size: 'small' })).toBe(2);
    expect(hasActiveModelFilters({ family: 'llama' })).toBe(true);
    expect(hasActiveModelFilters({})).toBe(false);
  });
});
