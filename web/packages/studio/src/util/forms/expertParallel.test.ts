// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  expertParallelSizeOnModelChange,
  expertParallelWarning,
  suggestedExpertParallelSize,
  validExpertParallelSizes,
} from '@studio/util/forms/expertParallel';

describe('validExpertParallelSizes', () => {
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

const spec = (checkpointModelName: string, numExperts?: number) =>
  ({
    checkpoint_model_name: checkpointModelName,
    moe_config: numExperts
      ? { num_experts: numExperts, num_experts_per_tok: 6, num_expert_layers: 23 }
      : undefined,
  }) as ModelEntity['spec'];

const nemotronH = spec('NemotronHForCausalLM', 128);

describe('expertParallelWarning', () => {
  it('accepts a valid size', () => {
    expect(
      expertParallelWarning(nemotronH, { num_gpus_per_node: 4, expert_parallel_size: 4 })
    ).toBeUndefined();
  });

  it.each([undefined, 1, 3])(
    'states the valid sizes for Nemotron-H when the size is %s',
    (size) => {
      expect(
        expertParallelWarning(nemotronH, { num_gpus_per_node: 4, expert_parallel_size: size })
      ).toBe('This model needs an expert parallel size of 2 or 4 to train on 4 GPUs.');
    }
  );

  it('recommends the valid sizes for other MoE architectures', () => {
    expect(expertParallelWarning(spec('MixtralForCausalLM', 8), { num_gpus_per_node: 4 })).toBe(
      'Mixture-of-Experts models usually need an expert parallel size of 2 or 4 to train on 4 GPUs.'
    );
  });

  it('warns about tensor parallelism', () => {
    expect(
      expertParallelWarning(nemotronH, {
        num_gpus_per_node: 4,
        tensor_parallel_size: 2,
        expert_parallel_size: 2,
      })
    ).toBe('This model needs Tensor Parallel Size set to 1.');
  });

  it('explains when no size fits', () => {
    expect(expertParallelWarning(nemotronH, { num_gpus_per_node: 3 })).toMatch(
      /no expert parallel size fits 3 GPUs/
    );
  });

  it.each([
    ['a single GPU', nemotronH, { num_gpus_per_node: 1 }],
    ['a dense model', spec('NemotronHForCausalLM'), { num_gpus_per_node: 4 }],
  ])('does not warn on %s', (_, modelSpec, p) => {
    expect(expertParallelWarning(modelSpec, p)).toBeUndefined();
  });
});

describe('suggestedExpertParallelSize', () => {
  it.each([undefined, 1, 3, 8])('suggests the largest valid size when the size is %s', (size) => {
    expect(
      suggestedExpertParallelSize(nemotronH, { num_gpus_per_node: 4, expert_parallel_size: size })
    ).toBe(4);
  });

  it('keeps a valid size', () => {
    expect(
      suggestedExpertParallelSize(nemotronH, { num_gpus_per_node: 4, expert_parallel_size: 2 })
    ).toBeUndefined();
  });

  it.each([
    ['a single GPU', nemotronH, { num_gpus_per_node: 1 }],
    ['a dense model', spec('NemotronHForCausalLM'), { num_gpus_per_node: 4 }],
    ['another MoE architecture', spec('MixtralForCausalLM', 8), { num_gpus_per_node: 4 }],
    ['a GPU count no size fits', nemotronH, { num_gpus_per_node: 3 }],
  ])('suggests nothing for %s', (_, modelSpec, p) => {
    expect(suggestedExpertParallelSize(modelSpec, p)).toBeUndefined();
  });

  it.each([
    ['a single GPU', nemotronH, { num_gpus_per_node: 1, expert_parallel_size: 4 }],
    ['a GPU count no size fits', nemotronH, { num_gpus_per_node: 3, expert_parallel_size: 4 }],
    ['a dense model', spec('LlamaForCausalLM'), { num_gpus_per_node: 4, expert_parallel_size: 4 }],
  ])('clears a set size on %s', (_, modelSpec, p) => {
    expect(suggestedExpertParallelSize(modelSpec, p)).toBeNull();
  });

  it('keeps a size set for another MoE architecture', () => {
    expect(
      suggestedExpertParallelSize(spec('MixtralForCausalLM', 8), {
        num_gpus_per_node: 4,
        expert_parallel_size: 2,
      })
    ).toBeUndefined();
  });
});

describe('expertParallelSizeOnModelChange', () => {
  it('clears a carried-over size for another MoE architecture', () => {
    expect(
      expertParallelSizeOnModelChange(spec('MixtralForCausalLM', 8), {
        num_gpus_per_node: 4,
        expert_parallel_size: 4,
      })
    ).toBeNull();
  });

  it('keeps a valid size for Nemotron-H', () => {
    expect(
      expertParallelSizeOnModelChange(nemotronH, { num_gpus_per_node: 4, expert_parallel_size: 2 })
    ).toBeUndefined();
  });
});
