// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixSecretResponse } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

// Mock secret data
const mockSecrets: HelixSecretResponse[] = [
  {
    name: 'openai-api-key',
    workspace: 'default',
    description: 'OpenAI API key for GPT-4 integration',
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-01-20T14:45:00Z',
  },
  {
    name: 'anthropic-api-key',
    workspace: 'default',
    description: 'Anthropic API key for Claude integration',
    created_at: '2024-01-18T09:15:00Z',
    updated_at: '2024-01-18T09:15:00Z',
  },
  {
    name: 'huggingface-token',
    workspace: 'default',
    description: 'HuggingFace access token for model downloads',
    created_at: '2024-01-10T16:20:00Z',
    updated_at: '2024-01-25T11:30:00Z',
  },
];

export const secretsHandlers = [
  http.get(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`, () =>
    HttpResponse.json({
      data: mockSecrets,
      pagination: {
        page: 1,
        page_size: 25,
        current_page_size: mockSecrets.length,
        total_pages: 1,
        total_results: mockSecrets.length,
      },
    })
  ),
  http.post(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets`, () =>
    HttpResponse.json({
      name: 'test-secret',
      workspace: 'default',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  ),
  http.get(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets/:name`, () =>
    HttpResponse.json({
      name: 'test-secret',
      workspace: 'default',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  ),
  http.patch(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets/:name`, () =>
    HttpResponse.json({
      name: 'test-secret',
      workspace: 'default',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),
  http.get(`${PLATFORM_BASE_URL}/apis/secrets/v2/workspaces/:workspace/secrets/:name/access`, () =>
    HttpResponse.json({
      value: 'secret-value',
    })
  ),
];
