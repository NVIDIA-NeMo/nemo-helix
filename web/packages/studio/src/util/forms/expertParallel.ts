// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AutomodelParallelismSpec } from '@nemo/sdk/generated/customizer/schema/AutomodelParallelismSpec';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

/**
 * The expert count of a model whose Automodel implementation needs expert parallelism on more
 * than one GPU, or `undefined` for any other model. The training container requires this for
 * every MoE architecture with a custom Automodel implementation, but the model entity does not
 * say which those are, so only Nemotron-H is recognised here.
 */
export const expertParallelExpertCount = (model: ModelEntity | undefined): number | undefined => {
  const spec = model?.spec;
  const numExperts = spec?.moe_config?.num_experts;
  return spec?.checkpoint_model_name === 'NemotronHForCausalLM' && numExperts && numExperts > 1
    ? numExperts
    : undefined;
};

const totalGpus = (p: AutomodelParallelismSpec) => (p.num_nodes ?? 1) * (p.num_gpus_per_node ?? 1);

/**
 * Expert parallel sizes the run accepts, or `undefined` when it does not need expert
 * parallelism. EP shards experts across the data × context parallel ranks, so it must divide
 * both that group and the expert count.
 */
export const validExpertParallelSizes = (
  numExperts: number | undefined,
  p: AutomodelParallelismSpec
): number[] | undefined => {
  if (!numExperts || totalGpus(p) <= 1) return undefined;
  const cp = p.context_parallel_size ?? 1;
  const modelParallel = (p.tensor_parallel_size ?? 1) * (p.pipeline_parallel_size ?? 1) * cp;
  const dpCp = Math.floor(totalGpus(p) / modelParallel) * cp;
  const sizes: number[] = [];
  for (let ep = 2; ep <= dpCp; ep++) {
    if (dpCp % ep === 0 && numExperts % ep === 0) sizes.push(ep);
  }
  return sizes;
};

/** Why the run's expert parallelism would be rejected, or `undefined` when it is fine. */
export const expertParallelError = (
  numExperts: number | undefined,
  p: AutomodelParallelismSpec
): string | undefined => {
  const sizes = validExpertParallelSizes(numExperts, p);
  if (!sizes) return undefined;
  if ((p.tensor_parallel_size ?? 1) > 1) {
    return 'This Mixture-of-Experts model does not support tensor parallelism. Set Tensor Parallel Size to 1.';
  }
  if (sizes.length === 0) {
    return `No expert parallel size fits ${totalGpus(p)} GPUs for this model's ${numExperts} experts. Change the GPU count or pipeline and context parallelism.`;
  }
  const ep = p.expert_parallel_size;
  if (ep === undefined || !sizes.includes(ep)) {
    return `With ${totalGpus(p)} GPUs, this Mixture-of-Experts model needs one of ${sizes.join(', ')}.`;
  }
  return undefined;
};
