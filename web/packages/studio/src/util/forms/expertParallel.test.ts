// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
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

describe('expertParallelWarning', () => {
  it('accepts a valid size', () => {
    expect(
      expertParallelWarning(128, { num_gpus_per_node: 4, expert_parallel_size: 4 })
    ).toBeUndefined();
  });

  it.each([undefined, 1, 3])('lists the valid sizes when the size is %s', (size) => {
    expect(expertParallelWarning(128, { num_gpus_per_node: 4, expert_parallel_size: size })).toBe(
      'This Mixture-of-Experts model will likely fail to train on 4 GPUs without an expert parallel size of 2, 4.'
    );
  });

  it('warns about tensor parallelism', () => {
    expect(
      expertParallelWarning(128, {
        num_gpus_per_node: 4,
        tensor_parallel_size: 2,
        expert_parallel_size: 2,
      })
    ).toMatch(/tensor parallelism/);
  });

  it('explains when no size fits', () => {
    expect(expertParallelWarning(128, { num_gpus_per_node: 3 })).toMatch(
      /No expert parallel size fits 3 GPUs/
    );
  });

  it.each([
    ['a single GPU', 128, { num_gpus_per_node: 1 }],
    ['a dense model', undefined, { num_gpus_per_node: 4 }],
  ])('does not warn on %s', (_, numExperts, p) => {
    expect(expertParallelWarning(numExperts, p)).toBeUndefined();
  });
});

describe('suggestedExpertParallelSize', () => {
  it.each([undefined, 1, 3, 8])('suggests the largest valid size when the size is %s', (size) => {
    expect(
      suggestedExpertParallelSize(128, { num_gpus_per_node: 4, expert_parallel_size: size })
    ).toBe(4);
  });

  it('keeps a valid size', () => {
    expect(
      suggestedExpertParallelSize(128, { num_gpus_per_node: 4, expert_parallel_size: 2 })
    ).toBeUndefined();
  });

  it.each([
    ['a single GPU', 128, { num_gpus_per_node: 1 }],
    ['a dense model', undefined, { num_gpus_per_node: 4 }],
    ['a GPU count no size fits', 128, { num_gpus_per_node: 3 }],
  ])('suggests nothing on %s', (_, numExperts, p) => {
    expect(suggestedExpertParallelSize(numExperts, p)).toBeUndefined();
  });
});
