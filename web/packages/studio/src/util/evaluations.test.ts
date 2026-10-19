// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  buildModelPayload,
  getModelName,
  parseEvaluationModelValue,
  prettifyName,
} from '@studio/util/evaluations';

const makeModel = (
  overrides: Partial<ModelEntity> & Pick<ModelEntity, 'name' | 'workspace'>
): ModelEntity =>
  ({
    id: 'test-id',
    created_at: '',
    updated_at: '',
    ...overrides,
  }) as ModelEntity;

describe('evaluation utils', () => {
  describe('getModelName', () => {
    it('should return the model name when the target has a name', () => {
      const job = { spec: { target: { name: 'meta/llama-3.1-8b-instruct' } } };
      expect(getModelName(job as never)).toBe('meta/llama-3.1-8b-instruct');
    });

    it('should return the model name when the target is an agent with a name', () => {
      const job = { spec: { target: { name: 'my-model' } } };
      expect(getModelName(job as never)).toBe('my-model');
    });

    it('should return N/A when no model is found', () => {
      const job = { spec: {} };
      expect(getModelName(job as never)).toBe('N/A');
    });

    it('should return N/A when row is undefined', () => {
      expect(getModelName(undefined)).toBe('N/A');
    });
  });

  describe('prettifyName', () => {
    it('should convert snake_case to Title Case and remove Score suffix', () => {
      expect(prettifyName('accuracy_score')).toBe('Accuracy');
      expect(prettifyName('f1_score')).toBe('F1');
    });

    it('should handle names without Score suffix', () => {
      expect(prettifyName('rouge_metric')).toBe('Rouge Metric');
    });
  });
});

describe('parseEvaluationModelValue', () => {
  it('returns the full string as modelUrn when no adapter delimiter is present', () => {
    expect(parseEvaluationModelValue('default/llama-3.1-8b')).toEqual({
      modelUrn: 'default/llama-3.1-8b',
      adapterName: null,
    });
  });

  it('splits on the first :: to extract modelUrn and adapterName', () => {
    expect(parseEvaluationModelValue('default/llama-3.1-8b::my-lora')).toEqual({
      modelUrn: 'default/llama-3.1-8b',
      adapterName: 'my-lora',
    });
  });

  it('handles adapter names that contain colons', () => {
    expect(parseEvaluationModelValue('ws/model::adapter::v2')).toEqual({
      modelUrn: 'ws/model',
      adapterName: 'adapter::v2',
    });
  });
});

describe('buildModelPayload', () => {
  const origin = 'https://nhx.example.com';

  const models: ModelEntity[] = [
    makeModel({
      name: 'llama-3.1-8b',
      workspace: 'default',
      model_providers: ['default/nim-llama'],
      adapters: [
        {
          name: 'my-lora',
          fileset: 'default/lora-files',
          finetuning_type: 'LORA' as never,
          enabled: true,
          workspace: 'default',
        },
      ],
    }),
    makeModel({
      name: 'mistral-7b',
      workspace: 'team-a',
      model_providers: ['nim-mistral'],
    }),
  ];

  it('returns the value string as-is for a base model (no adapter)', () => {
    const result = buildModelPayload('default/llama-3.1-8b', models, origin);
    expect(result).toEqual({ ok: true, payload: 'default/llama-3.1-8b' });
  });

  it('builds an EvaluatorModel with the provider proxy URL for an adapter', () => {
    const result = buildModelPayload('default/llama-3.1-8b::my-lora', models, origin);
    expect(result).toEqual({
      ok: true,
      payload: {
        url: 'https://nhx.example.com/apis/inference-gateway/v2/workspaces/default/provider/nim-llama/-/v1',
        name: 'my-lora',
      },
    });
  });

  it('strips the workspace prefix from a qualified provider ref', () => {
    const result = buildModelPayload('default/llama-3.1-8b::my-lora', models, origin);
    expect(result).toEqual(
      expect.objectContaining({
        ok: true,
        payload: expect.objectContaining({
          url: expect.stringContaining('/provider/nim-llama/'),
        }),
      })
    );
  });

  it('handles a provider ref without a workspace prefix', () => {
    const result = buildModelPayload('team-a/mistral-7b::some-adapter', models, origin);
    expect(result).toEqual({
      ok: true,
      payload: {
        url: 'https://nhx.example.com/apis/inference-gateway/v2/workspaces/team-a/provider/nim-mistral/-/v1',
        name: 'some-adapter',
      },
    });
  });

  it('returns an error when the parent model is not found', () => {
    const result = buildModelPayload('unknown/model::adapter', models, origin);
    expect(result).toEqual({
      ok: false,
      error: 'Selected model not found or still loading.',
    });
  });

  it('returns an error when the parent model has no providers', () => {
    const noProviderModels = [makeModel({ name: 'bare-model', workspace: 'ws' })];
    const result = buildModelPayload('ws/bare-model::adapter', noProviderModels, origin);
    expect(result).toEqual({
      ok: false,
      error: 'Selected model has no provider configured for adapter inference.',
    });
  });

  it('returns an error when model_providers is an empty array', () => {
    const emptyProviderModels = [
      makeModel({ name: 'bare-model', workspace: 'ws', model_providers: [] }),
    ];
    const result = buildModelPayload('ws/bare-model::adapter', emptyProviderModels, origin);
    expect(result).toEqual({
      ok: false,
      error: 'Selected model has no provider configured for adapter inference.',
    });
  });
});
