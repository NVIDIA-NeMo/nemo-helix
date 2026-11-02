// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AutomodelParallelismSpec } from '@nemo/sdk/generated/customizer/schema/AutomodelParallelismSpec';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

const totalGpus = (p: AutomodelParallelismSpec) => (p.num_nodes ?? 1) * (p.num_gpus_per_node ?? 1);

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

// Only Nemotron-H is confirmed to need expert parallelism. Automodel rejects it for MoE models on
// its Hugging Face path, such as Mixtral, so it is only filled in for Nemotron-H.
const isNemotronH = (spec: ModelEntity['spec']) =>
  spec?.checkpoint_model_name === 'NemotronHForCausalLM';

export const expertParallelWarning = (
  spec: ModelEntity['spec'],
  p: AutomodelParallelismSpec
): string | undefined => {
  const numExperts = spec?.moe_config?.num_experts;
  if (!numExperts || numExperts <= 1 || totalGpus(p) <= 1) return undefined;
  const needs = isNemotronH(spec) ? 'This model needs' : 'Mixture-of-Experts models usually need';
  if ((p.tensor_parallel_size ?? 1) > 1) return `${needs} Tensor Parallel Size set to 1.`;
  const sizes = validExpertParallelSizes(numExperts, p);
  if (sizes.length === 0) {
    return `${needs} expert parallelism on more than one GPU, but no expert parallel size fits ${totalGpus(p)} GPUs and ${numExperts} experts.`;
  }
  const ep = p.expert_parallel_size;
  if (ep === undefined || !sizes.includes(ep)) {
    return `${needs} an expert parallel size of ${sizes.join(' or ')} to train on ${totalGpus(p)} GPUs.`;
  }
  return undefined;
};

export const suggestedExpertParallelSize = (
  spec: ModelEntity['spec'],
  p: AutomodelParallelismSpec
): number | undefined => {
  const numExperts = spec?.moe_config?.num_experts;
  if (!isNemotronH(spec) || !numExperts || numExperts <= 1 || totalGpus(p) <= 1) return undefined;
  const sizes = validExpertParallelSizes(numExperts, p);
  return sizes.includes(p.expert_parallel_size ?? 0) ? undefined : sizes.at(-1);
};
