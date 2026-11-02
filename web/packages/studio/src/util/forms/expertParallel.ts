// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AutomodelParallelismSpec } from '@nemo/sdk/generated/customizer/schema/AutomodelParallelismSpec';

const totalGpus = (p: AutomodelParallelismSpec) => (p.num_nodes ?? 1) * (p.num_gpus_per_node ?? 1);

/** Expert parallel sizes above 1 that divide both data × context parallel size and the expert count. */
export const validExpertParallelSizes = (
  numExperts: number,
  p: AutomodelParallelismSpec
): number[] => {
  const cp = p.context_parallel_size ?? 1;
  const modelParallel = (p.tensor_parallel_size ?? 1) * (p.pipeline_parallel_size ?? 1) * cp;
  const dpCp = Math.floor(totalGpus(p) / modelParallel) * cp;
  const sizes: number[] = [];
  for (let ep = 2; ep <= dpCp; ep++) {
    if (dpCp % ep === 0 && numExperts % ep === 0) sizes.push(ep);
  }
  return sizes;
};

/** Why multi-GPU training of a Mixture-of-Experts model is likely to fail, or `undefined`. */
export const expertParallelWarning = (
  numExperts: number | undefined,
  p: AutomodelParallelismSpec
): string | undefined => {
  if (!numExperts || numExperts <= 1 || totalGpus(p) <= 1) return undefined;
  if ((p.tensor_parallel_size ?? 1) > 1) {
    return 'Mixture-of-Experts models usually do not support tensor parallelism. Set Tensor Parallel Size to 1.';
  }
  const sizes = validExpertParallelSizes(numExperts, p);
  if (sizes.length === 0) {
    return `No expert parallel size fits ${totalGpus(p)} GPUs for this model's ${numExperts} experts, so training will likely fail.`;
  }
  const ep = p.expert_parallel_size;
  if (ep === undefined || !sizes.includes(ep)) {
    return `This Mixture-of-Experts model will likely fail to train on ${totalGpus(p)} GPUs without an expert parallel size of ${sizes.join(', ')}.`;
  }
  return undefined;
};
