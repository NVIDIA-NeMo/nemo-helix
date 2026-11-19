// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomizationCreateAutomodelJobBody } from '@nemo/sdk/generated/customizer/zod/automodel-jobs';
import { CustomizationCreateRlJobBody } from '@nemo/sdk/generated/customizer/zod/rl-jobs';
import { CustomizationCreateUnslothJobBody } from '@nemo/sdk/generated/customizer/zod/unsloth-jobs';
import {
  AUTOMODEL_DEFAULT_SPEC,
  AUTOMODEL_SEED,
  AUTOMODEL_SPEC_DEFAULTS,
  DPO_SPEC_DEFAULTS,
  GRPO_SPEC_DEFAULTS,
  RL_DPO_DEFAULT_SPEC,
  RL_GRPO_DEFAULT_SPEC,
  UNSET_PLACEHOLDER,
  UNSLOTH_SEED,
  UNSLOTH_DEFAULT_SPEC,
  UNSLOTH_SPEC_DEFAULTS,
  booleanDefault,
  hasLearningRate,
  learningRateDefault,
  learningRateSliderProps,
  numberDefault,
  placeholderFor,
  resolveAutomodelRecipe,
  rlSeed,
  stringDefault,
  withoutLearningRate,
} from '@studio/util/forms/specDefaults';

/**
 * These assert against values that live in the OpenAPI spec, not in the form. If the
 * backend changes one of them the test fails, which is the point: it says the form is
 * now showing something different, rather than letting a stale copy sit unnoticed.
 */
describe('specDefaults', () => {
  it('reads scalar defaults out of the generated SDK', () => {
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'max_seq_length')).toBe(2048);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'batch_size')).toBe(32);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'learning_rate')).toBe(0.000005);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'adam_beta1')).toBe(0.9);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'keep_top_k')).toBe(1);
  });

  it('decodes multi-word and digit-suffixed field names', () => {
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'adam_beta2')).toBe(0.999);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'dynamic_sampling_max_gen_batches')).toBe(10);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'vllm_gpu_memory_utilization')).toBe(0.5);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'sequence_length_round')).toBe(64);
  });

  it('reads booleans and enums', () => {
    expect(booleanDefault(GRPO_SPEC_DEFAULTS, 'use_leave_one_out_baseline')).toBe(true);
    expect(booleanDefault(GRPO_SPEC_DEFAULTS, 'use_dynamic_sampling')).toBe(false);
    expect(booleanDefault(GRPO_SPEC_DEFAULTS, 'normalize_rewards')).toBe(true);
    expect(stringDefault(GRPO_SPEC_DEFAULTS, 'batching_strategy')).toBe('dynamic');
  });

  it('strips the branch marker from nullable object paths', () => {
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'lora_rank')).toBe(16);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'lora_alpha')).toBe(32);
  });

  /** The whole reason the two arms are separate tables. */
  it('keeps the DPO and GRPO union arms apart', () => {
    expect(numberDefault(DPO_SPEC_DEFAULTS, 'ref_policy_kl_penalty')).toBe(0.05);
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'ref_policy_kl_penalty')).toBe(0);
  });

  it('reports undefined for fields the spec leaves without a default', () => {
    // Generated SDK still has no default until it is regenerated from OpenAPI,
    // where `policy_backend` now defaults to `automodel`.
    for (const field of [
      'seed',
      'max_steps',
      'val_check_interval',
      'min_learning_rate',
      'policy_backend',
    ]) {
      expect(numberDefault(GRPO_SPEC_DEFAULTS, field)).toBeUndefined();
    }
  });

  it('placeholders show the backend default, or Unset when there is none', () => {
    expect(placeholderFor(GRPO_SPEC_DEFAULTS, 'max_seq_length')).toBe('2048');
    expect(placeholderFor(GRPO_SPEC_DEFAULTS, 'seed')).toBe(UNSET_PLACEHOLDER);
  });

  /** A generator rename would empty these maps; fail loudly rather than silently unset everything. */
  it('every backend table is populated', () => {
    expect(DPO_SPEC_DEFAULTS.size).toBeGreaterThan(10);
    expect(GRPO_SPEC_DEFAULTS.size).toBeGreaterThan(20);
    expect(AUTOMODEL_SPEC_DEFAULTS.size).toBeGreaterThan(5);
    expect(UNSLOTH_SPEC_DEFAULTS.size).toBeGreaterThan(5);
  });

  it('rejects a value of the wrong type instead of coercing it', () => {
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'policy_backend')).toBeUndefined();
    expect(booleanDefault(GRPO_SPEC_DEFAULTS, 'batch_size')).toBeUndefined();
  });

  /**
   * The module parses each seed at import time, and `.parse` throws. A backend adding a
   * required field the seeds do not supply would take the whole customization form down
   * rather than degrading to unset values, so assert the seeds still satisfy the schemas.
   */
  describe('seed parseability', () => {
    /**
     * The module parses each seed at import time, and `.parse` throws — a backend adding a
     * required field the seeds do not supply would take the whole customization form down
     * rather than degrading to unset values. Asserts the real seeds, not a hand-written
     * minimal spec, so a new required field fails here instead of at runtime.
     */
    it.each([
      ['automodel', CustomizationCreateAutomodelJobBody, AUTOMODEL_SEED],
      ['unsloth', CustomizationCreateUnslothJobBody, UNSLOTH_SEED],
      ['rl/dpo', CustomizationCreateRlJobBody, rlSeed('dpo')],
      ['rl/grpo', CustomizationCreateRlJobBody, rlSeed('grpo')],
    ])('%s seed still satisfies the generated schema', (_name, schema, seed) => {
      const result = schema.safeParse({ spec: seed });
      expect(result.success).toBe(true);
    });

    it('every table parsed to something usable', () => {
      for (const table of [
        AUTOMODEL_SPEC_DEFAULTS,
        UNSLOTH_SPEC_DEFAULTS,
        DPO_SPEC_DEFAULTS,
        GRPO_SPEC_DEFAULTS,
      ]) {
        expect(table.size).toBeGreaterThan(0);
      }
    });
  });
});

describe('learningRateDefault', () => {
  it('matches the backend validators for every selection', () => {
    expect(
      learningRateDefault({ backend: 'automodel', finetuningType: 'lora', recipe: 'auto' })
    ).toBe(5e-5);
    expect(
      learningRateDefault({ backend: 'automodel', finetuningType: 'all_weights', recipe: 'sft' })
    ).toBe(5e-6);
    expect(
      learningRateDefault({ backend: 'automodel', finetuningType: 'lora', recipe: 'bi_encoder' })
    ).toBe(1e-5);
    expect(
      learningRateDefault({
        backend: 'automodel',
        finetuningType: 'all_weights',
        recipe: 'cross_encoder',
      })
    ).toBe(3e-6);
    expect(learningRateDefault({ backend: 'unsloth', finetuningType: 'lora' })).toBe(2e-4);
    expect(learningRateDefault({ backend: 'unsloth', finetuningType: 'all_weights' })).toBe(2e-5);
    expect(learningRateDefault({ backend: 'rl', method: 'grpo', finetuningType: 'lora' })).toBe(
      1e-5
    );
    expect(
      learningRateDefault({ backend: 'rl', method: 'grpo', finetuningType: 'all_weights' })
    ).toBe(5e-6);
    expect(learningRateDefault({ backend: 'rl', method: 'dpo' })).toBe(5e-6);
  });

  it('takes the LoRA rate from the generated spec so a backend change to it is not shadowed', () => {
    expect(learningRateDefault({ backend: 'automodel', finetuningType: 'lora' })).toBe(
      numberDefault(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_learning_rate')
    );
    expect(learningRateDefault({ backend: 'unsloth', finetuningType: 'lora' })).toBe(
      numberDefault(UNSLOTH_SPEC_DEFAULTS, 'optimizer_learning_rate')
    );
  });
});

describe('unset learning rate', () => {
  it('leaves it out of every default spec', () => {
    expect(hasLearningRate('automodel', AUTOMODEL_DEFAULT_SPEC)).toBe(false);
    expect(hasLearningRate('unsloth', UNSLOTH_DEFAULT_SPEC)).toBe(false);
    expect(hasLearningRate('rl', RL_DPO_DEFAULT_SPEC)).toBe(false);
    expect(hasLearningRate('rl', RL_GRPO_DEFAULT_SPEC)).toBe(false);
  });

  it('keeps the rest of the group', () => {
    expect(AUTOMODEL_DEFAULT_SPEC.optimizer?.weight_decay).toBe(
      numberDefault(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_weight_decay')
    );
  });

  it('removes only the learning rate, without touching the input', () => {
    const spec = { optimizer: { learning_rate: 1e-4, weight_decay: 0.1 } };
    expect(withoutLearningRate('automodel', spec)).toEqual({ optimizer: { weight_decay: 0.1 } });
    expect(spec.optimizer.learning_rate).toBe(1e-4);
    expect(hasLearningRate('rl', { training: { learning_rate: 1e-5 } })).toBe(true);
  });

  it('shows the per-selection rate as the placeholder and resets to unset', () => {
    expect(
      learningRateSliderProps({ backend: 'automodel', finetuningType: 'all_weights' })
    ).toEqual({ defaultValue: undefined, unsetPlaceholder: String(5e-6), unsetValue: 5e-6 });
    expect(
      learningRateSliderProps({ backend: 'rl', method: 'grpo', finetuningType: 'lora' })
    ).toMatchObject({ defaultValue: undefined, unsetValue: 1e-5 });
  });

  it('shows Unset while an auto recipe has not resolved', () => {
    expect(
      learningRateSliderProps({ backend: 'automodel', finetuningType: 'lora', recipe: null })
    ).toEqual({
      defaultValue: undefined,
      unsetPlaceholder: UNSET_PLACEHOLDER,
      unsetValue: undefined,
    });
  });
});

describe('resolveAutomodelRecipe', () => {
  it('follows the checkpoint head while the recipe is auto', () => {
    expect(resolveAutomodelRecipe('auto', { head_type: 'embedding' })).toBe('bi_encoder');
    expect(resolveAutomodelRecipe('auto', { head_type: 'cross_encoder' })).toBe('cross_encoder');
    expect(resolveAutomodelRecipe('auto', { head_type: 'causal_lm' })).toBe('auto');
  });

  it('falls back to is_embedding_model when the head is unknown or missing', () => {
    expect(resolveAutomodelRecipe('auto', { head_type: 'unknown', is_embedding_model: true })).toBe(
      'bi_encoder'
    );
    expect(resolveAutomodelRecipe('auto', { is_embedding_model: true })).toBe('bi_encoder');
    expect(resolveAutomodelRecipe('auto', { head_type: 'unknown' })).toBe('auto');
  });

  it('keeps an explicit recipe, and auto when the model is not loaded', () => {
    expect(resolveAutomodelRecipe('sft', { head_type: 'embedding' })).toBe('sft');
    expect(resolveAutomodelRecipe('auto', undefined)).toBe('auto');
  });

  it('gives an embedding checkpoint on auto its recipe rate', () => {
    expect(
      learningRateDefault({
        backend: 'automodel',
        finetuningType: 'lora',
        recipe: resolveAutomodelRecipe('auto', { head_type: 'embedding' }),
      })
    ).toBe(1e-5);
  });
});
