// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { UnslothOptimizerSpecOptim } from '@nemo/sdk/generated/customizer/schema';
import { UNSLOTH_GRADIENT_CHECKPOINTING_ITEMS } from '@studio/components/NewCustomizationForm/constants';
import { selectItems } from '@studio/util/forms/selectItems';

describe('selectItems', () => {
  it('shows the API value as the label by default', () => {
    expect(selectItems({ bf16: 'bf16', fp16: 'fp16' })).toEqual([
      { value: 'bf16', children: 'bf16' },
      { value: 'fp16', children: 'fp16' },
    ]);
  });

  it('covers every value of the generated enum', () => {
    const values = selectItems(UnslothOptimizerSpecOptim).map((item) => item.value);
    expect(values).toEqual(Object.values(UnslothOptimizerSpecOptim));
  });

  it('applies an override only to the value it names', () => {
    const items = selectItems({ a: 'a', b: 'b' }, { a: 'Alpha' });
    expect(items).toEqual([
      { value: 'a', children: 'Alpha' },
      { value: 'b', children: 'b' },
    ]);
  });

  it('labels the gradient-checkpointing booleans rather than rendering them literally', () => {
    const labels = Object.fromEntries(
      UNSLOTH_GRADIENT_CHECKPOINTING_ITEMS.map((item) => [item.value, item.children])
    );
    expect(labels).toEqual({ unsloth: 'unsloth', true: 'Enabled', false: 'Disabled' });
  });
});
