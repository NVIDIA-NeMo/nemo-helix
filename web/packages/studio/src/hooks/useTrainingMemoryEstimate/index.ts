// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { modelsEstimateModelTrainingMemory } from '@nemo/sdk/generated/platform/models';
import type {
  TrainingMemoryEstimate,
  TrainingMemoryEstimateRequest,
} from '@nemo/sdk/generated/platform/schema';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { useQuery } from '@tanstack/react-query';
import { useDebounce } from 'use-debounce';

type AutomodelFields = CustomizationFormFields['automodel'];

const isPositiveInt = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value > 0;

/**
 * The estimate request for an automodel form, or null when the estimator does not model the run.
 *
 * The estimator covers causal-LM fine-tuning only: distillation also holds a teacher model, and
 * the embedding recipes train a different head. Fields still being typed (empty or zero) also
 * yield null rather than a request the API would reject.
 */
export const toTrainingMemoryEstimateRequest = (
  automodel: AutomodelFields
): TrainingMemoryEstimateRequest | null => {
  const { training, batch, parallelism } = automodel;
  if (training?.training_type === 'distillation') return null;
  if (training?.recipe === 'bi_encoder' || training?.recipe === 'cross_encoder') return null;

  const sizes = {
    max_seq_length: training?.max_seq_length,
    micro_batch_size: batch?.micro_batch_size,
    num_nodes: parallelism?.num_nodes ?? 1,
    num_gpus_per_node: parallelism?.num_gpus_per_node ?? 1,
    tensor_parallel_size: parallelism?.tensor_parallel_size ?? 1,
    pipeline_parallel_size: parallelism?.pipeline_parallel_size ?? 1,
    context_parallel_size: parallelism?.context_parallel_size ?? 1,
  };
  if (!Object.values(sizes).every(isPositiveInt)) return null;

  const usesLora =
    training?.finetuning_type === 'lora' || training?.finetuning_type === 'lora_merged';
  const expertParallelSize = parallelism?.expert_parallel_size;
  const packedSequenceSize = batch?.packed_sequence_size;

  return {
    ...(sizes as Record<keyof typeof sizes, number>),
    expert_parallel_size: isPositiveInt(expertParallelSize) ? expertParallelSize : undefined,
    lora_rank: usesLora ? (training?.lora?.rank ?? 16) : undefined,
    sequence_packing: batch?.sequence_packing ?? false,
    packed_sequence_size: isPositiveInt(packedSequenceSize) ? packedSequenceSize : undefined,
    activation_checkpointing: training?.activation_checkpointing ?? false,
  };
};

/**
 * Estimated per-GPU training memory for `modelRef` ("workspace/name") under `request`.
 *
 * Slider drags change the request on every step, so it is debounced before it reaches the API.
 * Failures (no model spec yet, a layout the GPU count cannot hold) resolve to no estimate: the
 * warning is advisory and must never stand in the way of submitting.
 */
export function useTrainingMemoryEstimate(
  modelRef: string | undefined,
  request: TrainingMemoryEstimateRequest | null
): TrainingMemoryEstimate | undefined {
  const [debouncedKey] = useDebounce(request ? JSON.stringify(request) : null, DEFAULT_DEBOUNCE_MS);
  const parts = modelRef ? getPartsFromReference(modelRef) : null;
  const enabled = Boolean(parts?.workspace && parts?.name && debouncedKey);

  const { data } = useQuery({
    queryKey: ['trainingMemoryEstimate', modelRef, debouncedKey],
    queryFn: ({ signal }) =>
      modelsEstimateModelTrainingMemory(
        parts?.workspace ?? '',
        parts?.name ?? '',
        JSON.parse(debouncedKey ?? '{}') as TrainingMemoryEstimateRequest,
        signal
      ),
    enabled,
    retry: false,
    // The estimate is a pure function of its inputs.
    staleTime: Infinity,
    // Hold the last estimate while a new one loads so the banner does not flicker mid-drag,
    // but never carry one model's estimate over to another.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === modelRef ? previous : undefined,
  });

  return enabled ? data : undefined;
}
