// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { buildModelConfig, getModelInferenceGatewayUrl, getModelTools } from '@studio/util/models';

vi.mock('@nemo/sdk/generated/platform/inference-gateway', () => ({
  getGatewayProxyGetQueryKey: (workspace: string, modelName: string, trailingUri: string) => [
    `/v2/nemo/workspaces/${workspace}/gateway/${modelName}/${trailingUri}`,
  ],
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return {
    ...actual,
    PLATFORM_BASE_URL: 'https://test.example.com',
  };
});

describe('getModelTools', () => {
  it('should parse tools from custom_fields string', () => {
    const tools = [{ type: 'function' as const, function: { name: 'test', parameters: {} } }];
    const model = { custom_fields: { tools: JSON.stringify(tools) } } as unknown as ModelEntity;
    expect(getModelTools(model)).toEqual(tools);
  });

  it('should return empty array when tools is not a string', () => {
    const model = { custom_fields: { tools: 123 } } as unknown as ModelEntity;
    expect(getModelTools(model)).toEqual([]);
  });

  it('should return empty array when custom_fields is undefined', () => {
    const model = {} as unknown as ModelEntity;
    expect(getModelTools(model)).toEqual([]);
  });

  it('should return empty array when tools is not present', () => {
    const model = { custom_fields: {} } as unknown as ModelEntity;
    expect(getModelTools(model)).toEqual([]);
  });
});

describe('getModelInferenceGatewayUrl', () => {
  it('should build chat completions URL', () => {
    const url = getModelInferenceGatewayUrl('my-ws', 'my-model');
    expect(url).toBe(
      'https://test.example.com/v2/nemo/workspaces/my-ws/gateway/my-model/v1/chat/completions'
    );
  });

  it('should build completions URL when isChat is false', () => {
    const url = getModelInferenceGatewayUrl('my-ws', 'my-model', false);
    expect(url).toBe(
      'https://test.example.com/v2/nemo/workspaces/my-ws/gateway/my-model/v1/completions'
    );
  });

  it('should strip workspace prefix from modelRef', () => {
    const url = getModelInferenceGatewayUrl('my-ws', 'my-ws/my-model');
    expect(url).toBe(
      'https://test.example.com/v2/nemo/workspaces/my-ws/gateway/my-model/v1/chat/completions'
    );
  });
});

describe('buildModelConfig', () => {
  it('should build chat config by default', () => {
    const config = buildModelConfig('urn:test/model');
    expect(config).toEqual({
      api_endpoint: {
        url: 'https://test.example.com/v1/chat/completions',
        model_id: 'urn:test/model',
        format: 'nim',
      },
    });
  });

  it('should build non-chat config when isChat is false', () => {
    const config = buildModelConfig('urn:test/model', false);
    expect(config).toEqual({
      api_endpoint: {
        url: 'https://test.example.com/v1/completions',
        model_id: 'urn:test/model',
        format: 'nim',
      },
    });
  });
});
