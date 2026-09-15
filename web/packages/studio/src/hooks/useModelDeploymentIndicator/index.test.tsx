// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { modelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import type {
  Adapter,
  ModelDeploymentStatus,
  ModelEntity,
  ModelProvider,
} from '@nemo/sdk/generated/platform/schema';
import { useModelDeploymentIndicator } from '@studio/hooks/useModelDeploymentIndicator';
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

vi.mock('@nemo/sdk/generated/platform/model-deployments', () => ({
  useModelsGetLatestDeployment: vi.fn(),
}));

const mockedGetProvider = vi.mocked(modelsGetProvider);
const mockedGetLatestDeployment = vi.mocked(useModelsGetLatestDeployment);

const BASE_ID = 'ws/base-model';
const ADAPTER_ID = 'ws/base-model&adapters/other-ws/my-adapter';

const model = {
  id: 'model-1',
  name: 'base-model',
  workspace: 'ws',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
  model_providers: ['ws/provider-a'],
} as ModelEntity;

const adapter = { name: 'my-adapter', workspace: 'other-ws' } as Adapter;

// `modelDeploymentId: null` means a provider that names no deployment (an external
// provider). It must not be `undefined`, which would re-apply the default below.
const buildProvider = (
  servedEntityIds: string[],
  modelDeploymentId: string | null = 'ws/dep-a'
): ModelProvider =>
  ({
    name: 'provider-a',
    workspace: 'ws',
    host_url: 'https://example.com',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    model_deployment_id: modelDeploymentId ?? undefined,
    served_models: servedEntityIds.map((id) => ({
      model_entity_id: id,
      served_model_name: id.replace(/\//g, '-'),
    })),
  }) as ModelProvider;

const mockDeployment = (status?: ModelDeploymentStatus, statusMessage?: string) => {
  mockedGetLatestDeployment.mockReturnValue({
    data: status ? { status, status_message: statusMessage } : undefined,
    isLoading: false,
  } as never);
};

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const renderIndicator = (a?: Adapter) =>
  renderHook(() => useModelDeploymentIndicator(model, a), { wrapper: createWrapper() });

describe('useModelDeploymentIndicator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDeployment('READY');
  });

  it('reports served for a model listed in served_models', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({ kind: 'served', status: 'READY' });
  });

  it('does not consult base_model to resolve a top-level row', async () => {
    // Regression: a row with no base_model used to fall through to "no deployment
    // found" even with a READY provider.
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    const { result } = renderHook(
      () => useModelDeploymentIndicator({ ...model, base_model: undefined }, undefined),
      { wrapper: createWrapper() }
    );
    await waitFor(() => expect(result.current.kind).toBe('served'));
  });

  it('reports not-deployed when nothing serves the model', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider(['ws/something-else']));
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('not-deployed'));
  });

  it('reports served for an adapter present under its composite id', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID, ADAPTER_ID]));
    const { result } = renderIndicator(adapter);
    await waitFor(() => expect(result.current.kind).toBe('served'));
  });

  it('reports adapter-not-loaded when the base is served but the adapter is not', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    const { result } = renderIndicator(adapter);
    await waitFor(() => expect(result.current.kind).toBe('adapter-not-loaded'));
  });

  it('reports not-deployed for an adapter whose base is not served either', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider(['ws/something-else']));
    const { result } = renderIndicator(adapter);
    await waitFor(() => expect(result.current.kind).toBe('not-deployed'));
  });

  it('surfaces a non-ready deployment status and message', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    mockDeployment('PENDING', 'Container running but not ready');
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({
      status: 'PENDING',
      statusMessage: 'Container running but not ready',
    });
  });

  it('treats a provider without a deployment as served with no status', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID], null));
    mockDeployment(undefined);
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({ status: undefined, providerRef: 'ws/provider-a' });
  });

  it('reports unknown, not not-deployed, when the provider fetch fails', async () => {
    // Provider queries do not retry, so one 5xx or auth failure means we never
    // learned what is served. Reporting "not deployed" there is confidently wrong.
    mockedGetProvider.mockRejectedValue(new Error('boom'));
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).not.toBe('loading'));
    expect(result.current.kind).toBe('unknown');
  });

  it('reports unknown for an adapter when one of several providers could not be read', async () => {
    // provider-a serves the base but not the adapter; provider-b fails. The adapter
    // may well be served by provider-b, so matching the base first and reporting
    // "Not served" would assert something we never actually learned.
    mockedGetProvider.mockImplementation((_workspace: string, name: string) =>
      name === 'provider-a'
        ? Promise.resolve(buildProvider([BASE_ID]))
        : Promise.reject(new Error('boom'))
    );
    const { result } = renderHook(
      () =>
        useModelDeploymentIndicator(
          { ...model, model_providers: ['ws/provider-a', 'ws/provider-b'] },
          adapter
        ),
      { wrapper: createWrapper() }
    );
    await waitFor(() => expect(result.current.kind).not.toBe('loading'));
    expect(result.current.kind).toBe('unknown');
  });

  it('still reports adapter-not-loaded when every provider was read', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    const { result } = renderHook(
      () =>
        useModelDeploymentIndicator(
          { ...model, model_providers: ['ws/provider-a', 'ws/provider-b'] },
          adapter
        ),
      { wrapper: createWrapper() }
    );
    await waitFor(() => expect(result.current.kind).not.toBe('loading'));
    expect(result.current.kind).toBe('adapter-not-loaded');
  });

  it('reports unknown for an adapter when the base probe fails', async () => {
    mockedGetProvider.mockRejectedValue(new Error('boom'));
    const { result } = renderIndicator(adapter);
    await waitFor(() => expect(result.current.kind).not.toBe('loading'));
    expect(result.current.kind).toBe('unknown');
  });

  it('marks hasDeployment false only when the provider names no deployment', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID], null));
    mockDeployment(undefined);
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({ hasDeployment: false });
  });

  it('keeps hasDeployment true when the deployment exists but its status is unreadable', async () => {
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID]));
    mockDeployment(undefined);
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({ hasDeployment: true, status: undefined });
  });

  it('reports not-deployed when the model has no providers', () => {
    const { result } = renderHook(
      () => useModelDeploymentIndicator({ ...model, model_providers: [] }, undefined),
      { wrapper: createWrapper() }
    );
    expect(result.current.kind).toBe('not-deployed');
  });
});
