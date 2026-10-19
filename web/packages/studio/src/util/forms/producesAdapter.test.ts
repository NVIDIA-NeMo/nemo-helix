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

  // Unsloth's equivalent of automodel's `lora_merged`, expressed at save time rather
  // than as a distinct finetuning_type. `finetuning_type` alone would call these
  // adapters and offer a base-model deployment for output that is full weights.
  describe('unsloth merges at save time', () => {
    const unsloth = (save_method?: string): FinetuningTypeSource => ({
      backend: 'unsloth',
      unsloth: { training: { finetuning_type: 'lora' }, output: { save_method } },
    });

    it.each(['merged_16bit', 'merged_4bit'])('is false for save_method=%s', (method) => {
      expect(producesAdapter(unsloth(method))).toBe(false);
    });

    it('is true for save_method=lora', () => {
      expect(producesAdapter(unsloth('lora'))).toBe(true);
    });

    // Omitted means the API default, which is `lora`.
    it('is true when save_method is unset', () => {
      expect(producesAdapter(unsloth(undefined))).toBe(true);
      expect(
        producesAdapter(
          fields({ backend: 'unsloth', unsloth: { training: { finetuning_type: 'lora' } } })
        )
      ).toBe(true);
    });

    // The merge happens at save time, so it has no bearing on the other backends —
    // and reading unsloth's namespace for an automodel run would be a cross-backend leak.
    it('does not let a stale unsloth save_method affect automodel', () => {
      expect(
        producesAdapter(
          fields({
            backend: 'automodel',
            automodel: { training: { finetuning_type: 'lora' } },
            unsloth: {
              training: { finetuning_type: 'lora' },
              output: { save_method: 'merged_16bit' },
            },
          })
        )
      ).toBe(true);
    });
  });
});
