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

const buildProvider = (
  servedEntityIds: string[],
  modelDeploymentId: string | undefined = 'ws/dep-a'
): ModelProvider =>
  ({
    name: 'provider-a',
    workspace: 'ws',
    host_url: 'https://example.com',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    model_deployment_id: modelDeploymentId,
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
    mockedGetProvider.mockResolvedValue(buildProvider([BASE_ID], undefined));
    mockDeployment(undefined);
    const { result } = renderIndicator();
    await waitFor(() => expect(result.current.kind).toBe('served'));
    expect(result.current).toMatchObject({ status: undefined, providerRef: 'ws/provider-a' });
  });

  it('reports not-deployed when the model has no providers', () => {
    const { result } = renderHook(
      () => useModelDeploymentIndicator({ ...model, model_providers: [] }, undefined),
      { wrapper: createWrapper() }
    );
    expect(result.current.kind).toBe('not-deployed');
  });
});
