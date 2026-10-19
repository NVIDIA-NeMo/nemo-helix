// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomizationCreateAutomodelJobBody } from '@nemo/sdk/generated/customizer/zod/automodel-jobs';
import { CustomizationCreateRlJobBody } from '@nemo/sdk/generated/customizer/zod/rl-jobs';
import { CustomizationCreateUnslothJobBody } from '@nemo/sdk/generated/customizer/zod/unsloth-jobs';
import {
  AUTOMODEL_SEED,
  AUTOMODEL_SPEC_DEFAULTS,
  DPO_SPEC_DEFAULTS,
  GRPO_SPEC_DEFAULTS,
  UNSET_PLACEHOLDER,
  UNSLOTH_SEED,
  UNSLOTH_SPEC_DEFAULTS,
  booleanDefault,
  numberDefault,
  placeholderFor,
  rlSeed,
  stringDefault,
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
    expect(numberDefault(GRPO_SPEC_DEFAULTS, 'learning_rate')).toBe(0.0001);
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
    // `policy_backend` is here rather than asserted as 'automodel': the backend derives it
    // from finetuning_type now, so the spec declares no default.
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
