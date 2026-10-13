// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useChatCompletion } from '@nemo/common/src/hooks/useChatCompletion';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';

const mocks = vi.hoisted(() => ({
  create: vi.fn(),
  user: {
    access_token: 'opaque-access-token',
    id_token:
      'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjQxMDI0NDQ4MDAsInN1YiI6InVzZXItMSJ9.signature',
    expired: false,
  },
}));

vi.mock('openai', () => ({
  default: class MockOpenAI {
    chat = { completions: { create: mocks.create } };
  },
}));

vi.mock('react-oidc-context', () => ({
  useAuth: () => ({ user: mocks.user }),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const request = {
  baseURL: 'http://localhost/v1',
  model: 'default/model',
  messages: [{ role: 'user' as const, content: 'Hello' }],
  stream: false as const,
};

describe('useChatCompletion authentication', () => {
  beforeEach(() => {
    mocks.create.mockReset();
    mocks.create.mockResolvedValue({ choices: [] });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('uses the access token by default', async () => {
    const { result } = renderHook(() => useChatCompletion(), { wrapper: createWrapper() });

    await result.current.mutateAsync(request);

    expect(mocks.create.mock.calls[0]?.[1]?.headers?.Authorization).toBe(
      'Bearer opaque-access-token'
    );
  });

  it('uses the ID token when configured', async () => {
    vi.stubEnv('VITE_AUTH_BEARER_TOKEN_SOURCE', 'id_token');
    const { result } = renderHook(() => useChatCompletion(), { wrapper: createWrapper() });

    await result.current.mutateAsync(request);

    expect(mocks.create.mock.calls[0]?.[1]?.headers?.Authorization).toBe(
      `Bearer ${mocks.user.id_token}`
    );
  });

  it('preserves an explicit token for an external endpoint', async () => {
    vi.stubEnv('VITE_AUTH_BEARER_TOKEN_SOURCE', 'id_token');
    const { result } = renderHook(() => useChatCompletion(), { wrapper: createWrapper() });

    await result.current.mutateAsync({ ...request, accessToken: 'external-service-token' });

    expect(mocks.create.mock.calls[0]?.[1]?.headers?.Authorization).toBe(
      'Bearer external-service-token'
    );
  });
});
