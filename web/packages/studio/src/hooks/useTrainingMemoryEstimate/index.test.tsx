// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TrainingMemoryEstimate } from '@nemo/sdk/generated/platform/schema';
import {
  toTrainingMemoryEstimateRequest,
  useTrainingMemoryEstimate,
} from '@studio/hooks/useTrainingMemoryEstimate';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { AUTOMODEL_DEFAULT_SPEC } from '@studio/util/forms/specDefaults';
import { renderHook, waitFor } from '@testing-library/react';

const mocks = vi.hoisted(() => ({ modelsEstimateModelTrainingMemory: vi.fn() }));

vi.mock('@nemo/sdk/generated/platform/models', () => ({
  modelsEstimateModelTrainingMemory: mocks.modelsEstimateModelTrainingMemory,
}));

type AutomodelFields = CustomizationFormFields['automodel'];

const automodel = (
  over: {
    training?: Partial<AutomodelFields['training']>;
    batch?: Partial<AutomodelFields['batch']>;
    parallelism?: Partial<AutomodelFields['parallelism']>;
  } = {}
): AutomodelFields => ({
  ...AUTOMODEL_DEFAULT_SPEC,
  model: 'ws/llama',
  training: { ...AUTOMODEL_DEFAULT_SPEC.training, max_seq_length: 4096, ...over.training },
  batch: { ...AUTOMODEL_DEFAULT_SPEC.batch, micro_batch_size: 2, ...over.batch },
  parallelism: {
    ...AUTOMODEL_DEFAULT_SPEC.parallelism,
    num_nodes: 1,
    num_gpus_per_node: 8,
    tensor_parallel_size: 2,
    ...over.parallelism,
  },
});

const ESTIMATE: TrainingMemoryEstimate = {
  gpu_count: 8,
  gpu_memory_gb: 80,
  gpu_memory_assumed: true,
  data_parallel_size: 4,
  model_states_gb: 52.4,
  activations_gb: 12.1,
  overhead_gb: 8.3,
  total_gb: 72.8,
  utilization: 0.91,
  warning_threshold: 0.85,
  status: 'near_capacity',
  limiting_factor: 'model_states',
};

describe('toTrainingMemoryEstimateRequest', () => {
  it('maps the automodel form fields onto the estimate request', () => {
    expect(
      toTrainingMemoryEstimateRequest(
        automodel({
          training: { finetuning_type: 'all_weights' },
          parallelism: { expert_parallel_size: 2 },
        })
      )
    ).toMatchObject({
      max_seq_length: 4096,
      micro_batch_size: 2,
      num_nodes: 1,
      num_gpus_per_node: 8,
      tensor_parallel_size: 2,
      expert_parallel_size: 2,
      lora_rank: undefined,
    });
  });

  it.each(['lora', 'lora_merged'] as const)('sends the LoRA rank for %s', (finetuningType) => {
    const request = toTrainingMemoryEstimateRequest(
      automodel({
        training: { finetuning_type: finetuningType, lora: { rank: 32 } },
      })
    );
    expect(request?.lora_rank).toBe(32);
  });

  it('passes packing and activation checkpointing through', () => {
    const request = toTrainingMemoryEstimateRequest(
      automodel({
        training: { activation_checkpointing: true },
        batch: { sequence_packing: true, packed_sequence_size: 8192 },
      })
    );
    expect(request).toMatchObject({
      sequence_packing: true,
      packed_sequence_size: 8192,
      activation_checkpointing: true,
    });
  });

  it('skips runs the estimator does not model', () => {
    expect(
      toTrainingMemoryEstimateRequest(automodel({ training: { training_type: 'distillation' } }))
    ).toBeNull();
    expect(
      toTrainingMemoryEstimateRequest(automodel({ training: { recipe: 'bi_encoder' } }))
    ).toBeNull();
  });

  it('skips a form with a size still being typed', () => {
    expect(
      toTrainingMemoryEstimateRequest(automodel({ batch: { micro_batch_size: 0 } }))
    ).toBeNull();
  });
});

describe('useTrainingMemoryEstimate', () => {
  beforeEach(() => {
    mocks.modelsEstimateModelTrainingMemory.mockReset();
  });

  it('fetches the estimate for the model the reference names', async () => {
    mocks.modelsEstimateModelTrainingMemory.mockResolvedValue(ESTIMATE);
    const request = toTrainingMemoryEstimateRequest(automodel());

    const { result } = renderHook(() => useTrainingMemoryEstimate('ws/llama', request), {
      wrapper,
    });

    await waitFor(() => expect(result.current).toEqual(ESTIMATE));
    expect(mocks.modelsEstimateModelTrainingMemory).toHaveBeenCalledWith(
      'ws',
      'llama',
      request,
      expect.anything()
    );
  });

  it('resolves to no estimate when the request fails', async () => {
    mocks.modelsEstimateModelTrainingMemory.mockRejectedValue(new Error('409'));

    const { result } = renderHook(
      () => useTrainingMemoryEstimate('ws/llama', toTrainingMemoryEstimateRequest(automodel())),
      { wrapper }
    );

    await waitFor(() => expect(mocks.modelsEstimateModelTrainingMemory).toHaveBeenCalled());
    expect(result.current).toBeUndefined();
  });

  it('does not call the API without a model or a request', async () => {
    renderHook(
      () => useTrainingMemoryEstimate(undefined, toTrainingMemoryEstimateRequest(automodel())),
      {
        wrapper,
      }
    );
    renderHook(() => useTrainingMemoryEstimate('ws/llama', null), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 600));
    expect(mocks.modelsEstimateModelTrainingMemory).not.toHaveBeenCalled();
  });
});
