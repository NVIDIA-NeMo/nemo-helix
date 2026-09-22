// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import type { ModelEntity, ModelProvider } from '@nemo/sdk/generated/platform/schema';
import { useModelIsServed } from '@studio/hooks/useModelIsServed';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import type { FC, PropsWithChildren } from 'react';

vi.mock('@nemo/sdk/generated/platform/model-providers', () => ({
  modelsGetProvider: vi.fn(),
  getModelsGetProviderQueryKey: (workspace: string, name: string) => [
    'models',
    'getProvider',
    workspace,
    name,
  ],
}));

const mockedGetProvider = vi.mocked(modelsGetProvider);

const buildModel = (overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({
    id: 'model-1',
    name: 'my-model',
    workspace: 'ws',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    model_providers: ['ws/provider-a'],
    ...overrides,
  }) as ModelEntity;

const buildProvider = (servedEntityIds: string[]): ModelProvider =>
  ({
    name: 'provider-a',
    workspace: 'ws',
    host_url: 'https://example.com',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    served_models: servedEntityIds.map((id) => ({
      model_entity_id: id,
      served_model_name: id.replace('/', '-'),
    })),
  }) as ModelProvider;

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useModelIsServed', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns isServed=false when model is undefined', () => {
    const { result } = renderHook(() => useModelIsServed(undefined), {
      wrapper: createWrapper(),
    });
    expect(result.current).toEqual({ isServed: false, isLoading: false, isError: false });
  });

  it('returns isServed=false when model has no providers', () => {
    const { result } = renderHook(() => useModelIsServed(buildModel({ model_providers: [] })), {
      wrapper: createWrapper(),
    });
    expect(result.current).toEqual({ isServed: false, isLoading: false, isError: false });
  });

  it('returns isServed=true when model is in provider served_models', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider(['ws/my-model', 'ws/other']));
    const { result } = renderHook(() => useModelIsServed(buildModel()), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isServed).toBe(true);
  });

  it('returns isServed=false when model is not in any provider served_models', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider(['ws/other-model']));
    const { result } = renderHook(() => useModelIsServed(buildModel()), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isServed).toBe(false);
  });

  it('checks all providers and returns true if any serves the model', async () => {
    mockedGetProvider
      .mockResolvedValueOnce(buildProvider(['ws/other-model']))
      .mockResolvedValueOnce(buildProvider(['ws/my-model']));
    const model = buildModel({ model_providers: ['ws/provider-a', 'ws/provider-b'] });
    const { result } = renderHook(() => useModelIsServed(model), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isServed).toBe(true);
  });

  // A failed lookup is not evidence that nothing serves the model. Callers that state
  // the negative affirmatively need to tell the two apart, so surface `isError`.
  it('returns isServed=false and flags isError when provider fetch fails', async () => {
    mockedGetProvider.mockRejectedValueOnce(new Error('network error'));
    const { result } = renderHook(() => useModelIsServed(buildModel()), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isServed).toBe(false);
    expect(result.current.isError).toBe(true);
  });

  it('does not flag isError when the lookup succeeds and simply finds nothing', async () => {
    mockedGetProvider.mockResolvedValueOnce(buildProvider(['ws/other-model']));
    const { result } = renderHook(() => useModelIsServed(buildModel()), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isServed).toBe(false);
    expect(result.current.isError).toBe(false);
  });
});
