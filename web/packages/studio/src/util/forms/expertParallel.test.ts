// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  expertParallelError,
  expertParallelExpertCount,
  validExpertParallelSizes,
} from '@studio/util/forms/expertParallel';

const model = (checkpointModelName: string, numExperts?: number) =>
  ({
    spec: {
      checkpoint_model_name: checkpointModelName,
      moe_config: numExperts
        ? { num_experts: numExperts, num_experts_per_tok: 6, num_expert_layers: 23 }
        : undefined,
    },
  }) as ModelEntity;

describe('expertParallelExpertCount', () => {
  it('returns the expert count for a Nemotron-H MoE model', () => {
    expect(expertParallelExpertCount(model('NemotronHForCausalLM', 128))).toBe(128);
  });

  it.each([
    ['a dense Nemotron-H model', model('NemotronHForCausalLM')],
    ['another MoE architecture', model('MixtralForCausalLM', 8)],
    ['a model that has not loaded', undefined],
  ])('ignores %s', (_, entity) => {
    expect(expertParallelExpertCount(entity)).toBeUndefined();
  });
});

describe('validExpertParallelSizes', () => {
  it('is undefined on a single GPU', () => {
    expect(validExpertParallelSizes(128, { num_gpus_per_node: 1 })).toBeUndefined();
  });

  it('is undefined for a model that does not need expert parallelism', () => {
    expect(validExpertParallelSizes(undefined, { num_gpus_per_node: 4 })).toBeUndefined();
  });

  it('lists the divisors of the GPU count that divide the expert count', () => {
    expect(validExpertParallelSizes(128, { num_gpus_per_node: 4 })).toEqual([2, 4]);
    expect(validExpertParallelSizes(6, { num_gpus_per_node: 4 })).toEqual([2]);
  });

  it('spans nodes', () => {
    expect(validExpertParallelSizes(128, { num_nodes: 2, num_gpus_per_node: 8 })).toEqual([
      2, 4, 8, 16,
    ]);
  });

  it('excludes GPUs taken by pipeline stages', () => {
    expect(
      validExpertParallelSizes(128, { num_gpus_per_node: 8, pipeline_parallel_size: 2 })
    ).toEqual([2, 4]);
  });
});

describe('expertParallelError', () => {
  it('accepts a valid size', () => {
    expect(
      expertParallelError(128, { num_gpus_per_node: 4, expert_parallel_size: 4 })
    ).toBeUndefined();
  });

  it.each([undefined, 1, 3])('lists the valid sizes when the size is %s', (size) => {
    expect(expertParallelError(128, { num_gpus_per_node: 4, expert_parallel_size: size })).toBe(
      'With 4 GPUs, expert parallel size must be one of 2, 4.'
    );
  });

  it('rejects tensor parallelism', () => {
    expect(
      expertParallelError(128, {
        num_gpus_per_node: 4,
        tensor_parallel_size: 2,
        expert_parallel_size: 2,
      })
    ).toMatch(/does not support tensor parallelism/);
  });

  it('explains when no size fits', () => {
    expect(expertParallelError(128, { num_gpus_per_node: 3 })).toMatch(
      /No expert parallel size fits 3 GPUs/
    );
  });

  it('does not apply on a single GPU', () => {
    expect(expertParallelError(128, { num_gpus_per_node: 1 })).toBeUndefined();
  });
});
