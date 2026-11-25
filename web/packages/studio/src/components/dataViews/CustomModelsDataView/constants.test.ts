// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FINETUNING_TYPE_FILTER_OPTIONS,
  toFinetuningTypeFilterState,
  toSelectedFinetuningTypes,
} from '@studio/components/dataViews/CustomModelsDataView/constants';

describe('FINETUNING_TYPE_FILTER_OPTIONS', () => {
  it('lists only product-supported finetuning types, deduped, with display labels', () => {
    expect(FINETUNING_TYPE_FILTER_OPTIONS).toEqual([
      { value: 'lora', label: 'LoRA' },
      { value: 'all_weights', label: 'All Weights' },
      { value: 'lora_merged', label: 'LoRA Merged' },
    ]);
  });

  it('excludes unsupported approaches the platform enum carries', () => {
    const values = FINETUNING_TYPE_FILTER_OPTIONS.map((option) => option.value);
    for (const unsupported of [
      'qlora',
      'dora',
      'prefix_tuning',
      'p_tuning',
      'soft_prompt',
      'ppo',
      'dpo',
      'grpo',
      'kto',
    ]) {
      expect(values).not.toContain(unsupported);
    }
  });
});

describe('finetuning type filter state', () => {
  it('reads selected types from filter state or a lone string from older URLs', () => {
    expect(toSelectedFinetuningTypes({ lora: true, dpo: true })).toEqual(['lora', 'dpo']);
    expect(toSelectedFinetuningTypes('dpo')).toEqual(['dpo']);
    expect(toSelectedFinetuningTypes(undefined)).toEqual([]);
  });

  it('builds filter state and clears it when nothing is selected', () => {
    expect(toFinetuningTypeFilterState(['lora'])).toEqual({ lora: true });
    expect(toFinetuningTypeFilterState([])).toBeUndefined();
  });
});
