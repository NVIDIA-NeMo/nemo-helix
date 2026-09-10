// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getActiveFinetuningType,
  producesAdapter,
  type FinetuningTypeSource,
} from '@studio/util/forms/customization';

const fields = (overrides: FinetuningTypeSource) => overrides;

describe('getActiveFinetuningType', () => {
  it('reads automodel from automodel.training', () => {
    expect(
      getActiveFinetuningType(
        fields({ backend: 'automodel', automodel: { training: { finetuning_type: 'lora' } } })
      )
    ).toBe('lora');
  });

  it('reads unsloth from unsloth.training', () => {
    expect(
      getActiveFinetuningType(
        fields({ backend: 'unsloth', unsloth: { training: { finetuning_type: 'all_weights' } } })
      )
    ).toBe('all_weights');
  });

  // RL keeps finetuning_type in the form-only `grpo` namespace, not on
  // `rl.training`, because one `rl.training` object is shared by DPO and GRPO.
  it('reads GRPO from the grpo namespace', () => {
    expect(
      getActiveFinetuningType(
        fields({ backend: 'rl', grpo: { trainingType: 'grpo', finetuning_type: 'lora' } })
      )
    ).toBe('lora');
  });

  it('returns undefined for DPO, which is always full-weight', () => {
    expect(
      getActiveFinetuningType(
        fields({ backend: 'rl', grpo: { trainingType: 'dpo', finetuning_type: 'lora' } })
      )
    ).toBeUndefined();
  });

  it('returns undefined when the backend branch is absent', () => {
    expect(getActiveFinetuningType(fields({ backend: 'automodel' }))).toBeUndefined();
  });
});

describe('producesAdapter', () => {
  it('is true for a plain LoRA run', () => {
    expect(
      producesAdapter(
        fields({ backend: 'automodel', automodel: { training: { finetuning_type: 'lora' } } })
      )
    ).toBe(true);
  });

  // The distinction this predicate exists for: lora_merged trains with LoRA but
  // merges into full weights, so its output is not a separately servable adapter.
  it('is false for lora_merged, whose output is full weights', () => {
    expect(
      producesAdapter(
        fields({
          backend: 'automodel',
          automodel: { training: { finetuning_type: 'lora_merged' } },
        })
      )
    ).toBe(false);
  });

  it('is false for all_weights', () => {
    expect(
      producesAdapter(
        fields({ backend: 'unsloth', unsloth: { training: { finetuning_type: 'all_weights' } } })
      )
    ).toBe(false);
  });

  it('is true for a GRPO LoRA run', () => {
    expect(
      producesAdapter(
        fields({ backend: 'rl', grpo: { trainingType: 'grpo', finetuning_type: 'lora' } })
      )
    ).toBe(true);
  });

  it('is false for DPO', () => {
    expect(producesAdapter(fields({ backend: 'rl', grpo: { trainingType: 'dpo' } }))).toBe(false);
  });
});
