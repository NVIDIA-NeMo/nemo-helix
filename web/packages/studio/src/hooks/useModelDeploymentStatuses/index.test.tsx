// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { modelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import type {
  Adapter,
  ModelDeployment,
  ModelEntity,
  ModelProvider,
} from '@nemo/sdk/generated/platform/schema';
import {
  useModelDeploymentStatuses,
  type DeploymentStatusTarget,
} from '@studio/hooks/useModelDeploymentStatuses';
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
  modelsGetLatestDeployment: vi.fn(),
  getModelsGetLatestDeploymentQueryKey: (workspace: string, name: string) => [
    'models',
    'getLatestDeployment',
    workspace,
    name,
  ],
}));

const mockedGetProvider = vi.mocked(modelsGetProvider);
const mockedGetDeployment = vi.mocked(modelsGetLatestDeployment);

const BASE = 'ws/base-model';
const COMPOSITE = 'ws/base-model&adapters/ws/my-adapter';

const model = {
  id: 'model-1',
  name: 'base-model',
  workspace: 'ws',
  model_providers: ['ws/provider-a'],
} as ModelEntity;

const adapter = { name: 'my-adapter', workspace: 'ws' } as Adapter;

const provider = (servedIds: string[], deploymentId: string | null = 'ws/dep-a'): ModelProvider =>
  ({
    name: 'provider-a',
    workspace: 'ws',
    model_deployment_id: deploymentId ?? undefined,
    served_models: servedIds.map((id) => ({ model_entity_id: id, served_model_name: id })),
  }) as ModelProvider;

const createWrapper = (): FC<PropsWithChildren> => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const renderStatuses = (targets: DeploymentStatusTarget[]) =>
  renderHook(() => useModelDeploymentStatuses(targets), { wrapper: createWrapper() });

describe('useModelDeploymentStatuses', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedGetDeployment.mockResolvedValue({ status: 'READY' } as ModelDeployment);
  });

  it('resolves a model and its adapters in one pass', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE, COMPOSITE]));
    const { result } = renderStatuses([
      { key: 'row-model', model },
      { key: 'row-adapter', model, adapter },
    ]);
    await waitFor(() => expect(result.current.get('row-model')?.kind).toBe('served'));
    expect(result.current.get('row-adapter')?.kind).toBe('served');
  });

  it('marks an adapter not-loaded when only its base is served', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE]));
    const { result } = renderStatuses([
      { key: 'row-model', model },
      { key: 'row-adapter', model, adapter },
    ]);
    await waitFor(() => expect(result.current.get('row-adapter')?.kind).toBe('adapter-not-loaded'));
    expect(result.current.get('row-model')?.kind).toBe('served');
  });

  it('reports not-deployed when a model has no providers', () => {
    const { result } = renderStatuses([
      { key: 'row', model: { ...model, model_providers: [] } as ModelEntity },
    ]);
    expect(result.current.get('row')?.kind).toBe('not-deployed');
  });

  it('reports unknown rather than not-deployed when a provider cannot be read', async () => {
    mockedGetProvider.mockRejectedValue(new Error('boom'));
    const { result } = renderStatuses([{ key: 'row', model }]);
    await waitFor(() => expect(result.current.get('row')?.kind).not.toBe('loading'));
    expect(result.current.get('row')?.kind).toBe('unknown');
  });

  it('keeps an adapter unknown when one of several providers cannot be read', async () => {
    // provider-a serves the base only; provider-b fails and may be serving the adapter.
    mockedGetProvider.mockImplementation((_ws: string, name: string) =>
      name === 'provider-a' ? Promise.resolve(provider([BASE])) : Promise.reject(new Error('boom'))
    );
    const multi = { ...model, model_providers: ['ws/provider-a', 'ws/provider-b'] } as ModelEntity;
    const { result } = renderStatuses([{ key: 'row', model: multi, adapter }]);
    await waitFor(() => expect(result.current.get('row')?.kind).not.toBe('loading'));
    expect(result.current.get('row')?.kind).toBe('unknown');
  });

  it('distinguishes an external provider from an unreadable deployment', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE], null));
    const { result } = renderStatuses([{ key: 'row', model }]);
    await waitFor(() => expect(result.current.get('row')?.kind).toBe('served'));
    expect(result.current.get('row')).toMatchObject({ hasDeployment: false });
  });

  it('does not consult base_model to resolve a row', async () => {
    // Regression: rows with no base_model -- every base model that hosts adapters --
    // used to resolve as undeployed even with a READY provider.
    mockedGetProvider.mockResolvedValue(provider([BASE]));
    const { result } = renderStatuses([
      { key: 'row', model: { ...model, base_model: undefined } as ModelEntity },
    ]);
    await waitFor(() => expect(result.current.get('row')?.kind).toBe('served'));
  });

  it('surfaces the deployment status and message', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE]));
    mockedGetDeployment.mockResolvedValue({
      status: 'PENDING',
      status_message: 'Container running but not ready',
    } as ModelDeployment);
    const { result } = renderStatuses([{ key: 'row', model }]);
    await waitFor(() => expect(result.current.get('row')?.kind).toBe('served'));
    expect(result.current.get('row')).toMatchObject({
      status: 'PENDING',
      statusMessage: 'Container running but not ready',
    });
  });

  it('keeps hasDeployment true when the deployment exists but its status is unreadable', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE]));
    mockedGetDeployment.mockRejectedValue(new Error('boom'));
    const { result } = renderStatuses([{ key: 'row', model }]);
    await waitFor(() => expect(result.current.get('row')?.kind).toBe('served'));
    expect(result.current.get('row')).toMatchObject({ hasDeployment: true, status: undefined });
  });

  it('returns a stable map identity across re-renders once resolved', async () => {
    // useQueries returns a fresh array every render, so without `combine` the
    // derived map would change identity every render -- which defeats the
    // useCallback around the table's makeColumns, since it depends on this map.
    mockedGetProvider.mockResolvedValue(provider([BASE]));
    const targets = [{ key: 'row', model }];
    const { result, rerender } = renderHook(() => useModelDeploymentStatuses(targets), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.get('row')?.kind).toBe('served'));
    const first = result.current;
    rerender();
    rerender();
    expect(result.current).toBe(first);
  });

  it('fetches each provider once even when many rows share it', async () => {
    mockedGetProvider.mockResolvedValue(provider([BASE, COMPOSITE]));
    const { result } = renderStatuses([
      { key: 'a', model },
      { key: 'b', model, adapter },
      { key: 'c', model },
    ]);
    await waitFor(() => expect(result.current.get('a')?.kind).toBe('served'));
    expect(mockedGetProvider).toHaveBeenCalledTimes(1);
  });
});
