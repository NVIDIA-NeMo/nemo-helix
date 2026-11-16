// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getModelMetadata,
  parametersToString,
} from '@nemo/common/src/components/ModelDetailsTooltip/utils';
import { MODEL_METADATA } from '@nemo/common/src/constants/modelMetadata';
import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { FinetuningType, ModelEntity } from '@nemo/sdk/generated/platform/schema';

// Mock the dependencies
vi.mock('@nemo/common/src/namedEntity');
vi.mock('@nemo/common/src/constants/modelMetadata');

const mockGetURNFromNamedEntityRef = vi.mocked(getURNFromNamedEntityRef);
const mockMODEL_METADATA = vi.mocked(MODEL_METADATA);

describe('parametersToString', () => {
  it('returns the number as string for values less than 1000', () => {
    expect(parametersToString(0)).toBe('0');
    expect(parametersToString(1)).toBe('1');
    expect(parametersToString(999)).toBe('999');
  });

  it('returns thousands for values between 1000 and 999999', () => {
    expect(parametersToString(1000)).toBe('1 thousand');
    expect(parametersToString(1500)).toBe('2 thousand');
  });

  it('returns millions for values between 1000000 and 999999999', () => {
    expect(parametersToString(1000000)).toBe('1 million');
    expect(parametersToString(1500000)).toBe('2 million');
  });

  it('returns billions for values between 1000000000 and 999999999999', () => {
    expect(parametersToString(1000000000)).toBe('1 billion');
    expect(parametersToString(1500000000)).toBe('2 billion');
  });

  it('returns trillions for values 1000000000000 and above', () => {
    expect(parametersToString(1_500_000_000_000)).toBe('2 trillion');
  });
  it('returns short format', () => {
    expect(parametersToString(1000, { format: 'short' })).toBe('1K');
    expect(parametersToString(1500, { format: 'short' })).toBe('2K');
    expect(parametersToString(1000000, { format: 'short' })).toBe('1M');
    expect(parametersToString(1500000, { format: 'short' })).toBe('2M');
    expect(parametersToString(1000000000, { format: 'short' })).toBe('1B');
    expect(parametersToString(1500000000, { format: 'short' })).toBe('2B');
    expect(parametersToString(1_500_000_000_000, { format: 'short' })).toBe('2T');
  });
});

describe('getModelMetadata', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockMODEL_METADATA['test/model'] = {
      name: 'Test Model',
      creator: 'Test Creator',
      architecture: 'transformer',
      description: 'Test description',
    };
  });

  it('returns undefined when modelId is not found in MODEL_METADATA', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('unknown/model@v1');

    const model = {
      id: 'test-id',
      name: 'Test Model',
      workspace: 'test-namespace',
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);
    expect(result).toBeUndefined();
  });

  it('returns base metadata when model has no target property', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Test Model',
      workspace: 'test-namespace',
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);
    expect(result).toEqual(mockMODEL_METADATA['test/model']);
  });

  it('returns base metadata when target is null or undefined', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Test Model',
      workspace: 'test-namespace',
      target: null,
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);
    expect(result).toEqual(mockMODEL_METADATA['test/model']);
  });

  it('handles ModelEntity with object target', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Custom Model',
      workspace: 'custom-namespace',
      target: {
        num_parameters: 70000000000,
      },
      training_options: [
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 16,
        },
        {
          finetuning_type: 'all_weights' as FinetuningType,
          num_gpus: 32,
        },
      ],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result).toEqual({
      ...mockMODEL_METADATA['test/model'],
      name: 'Custom Model',
      creator: 'custom-namespace',
      architecture: 'transformer',
      parameters: '70 billion',
      'fine-tune-options': ['lora', 'all_weights'],
      'recommended-gpus-for-customization': {
        lora: 16,
        all_weights: 32,
      },
    });
  });

  it('deduplicates fine-tune options when there are duplicates', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Custom Model',
      workspace: 'custom-namespace',
      target: {
        num_parameters: 1000000000,
      },
      training_options: [
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 8,
        },
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 16,
        },
        {
          finetuning_type: 'all_weights' as FinetuningType,
          num_gpus: 32,
        },
      ],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result?.['fine-tune-options']).toEqual(['lora', 'all_weights']);
    expect(result?.['recommended-gpus-for-customization']).toEqual({
      lora: 16, // Last value wins
      all_weights: 32,
    });
  });

  it('handles edge case with empty training_options array', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Custom Model',
      workspace: 'custom-namespace',
      target: {
        num_parameters: 1000000000,
      },
      training_options: [],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result?.['fine-tune-options']).toEqual([]);
    expect(result?.['recommended-gpus-for-customization']).toEqual({});
  });

  it('handles very large parameter counts correctly', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Custom Model',
      workspace: 'custom-namespace',
      target: {
        num_parameters: 1_000_000_000_000_000, // 1 quadrillion
      },
      training_options: [
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 64,
        },
      ],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result?.parameters).toBe('1000 trillion');
  });

  it('handles small parameter counts correctly', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: 'Custom Model',
      workspace: 'custom-namespace',
      target: {
        num_parameters: 500, // Less than 1000
      },
      training_options: [
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 1,
        },
      ],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result?.parameters).toBe('500');
  });

  it('handles null or undefined model properties gracefully', () => {
    mockGetURNFromNamedEntityRef.mockReturnValue('test/model@v1');

    const model = {
      id: 'test-id',
      name: null,
      namespace: null,
      target: {
        num_parameters: 1000000000,
      },
      training_options: [
        {
          finetuning_type: 'lora' as FinetuningType,
          num_gpus: 8,
        },
      ],
    } as unknown as ModelEntity;

    const result = getModelMetadata(model);

    expect(result?.name).toBeNull();
    expect(result?.creator).toBeNull();
    expect(result?.parameters).toBe('1 billion');
  });
});
