// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useModelSearch } from '@nemo/common/src/api/models/useModelSearch';
import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import type { ModelEntity, ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

vi.mock('@nemo/sdk/generated/platform/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/models')>();
  return { ...actual, modelsListModels: vi.fn() };
});

const mockProviders = vi.hoisted(() => ({
  data: [] as { workspace: string; name: string }[],
}));

vi.mock('@nemo/sdk/generated/platform/model-providers', () => ({
  useModelsListProviders: () => ({ data: { data: mockProviders.data }, isLoading: false }),
}));

const mockListModels = vi.mocked(modelsListModels);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const makeModel = (name: string, overrides: Partial<ModelEntity> = {}): ModelEntity =>
  ({ id: name, name, workspace: 'ws1', ...overrides }) as ModelEntity;

const makePage = (data: ModelEntity[], page: number, totalPages: number): ModelEntitysPage =>
  ({ data, pagination: { page, total_pages: totalPages } }) as ModelEntitysPage;

const renderSearch = (options: Partial<Parameters<typeof useModelSearch>[0]> = {}) =>
  renderHook(() => useModelSearch({ workspace: 'ws1', ...options }), { wrapper: createWrapper() });

beforeEach(() => {
  mockListModels.mockReset();
  mockProviders.data = [];
});

describe('useModelSearch', () => {
  it('stays idle while disabled', () => {
    renderSearch({ enabled: false });
    expect(mockListModels).not.toHaveBeenCalled();
  });

  it('stays idle without a workspace', () => {
    renderHook(() => useModelSearch({ workspace: null }), { wrapper: createWrapper() });
    expect(mockListModels).not.toHaveBeenCalled();
  });

  it('groups the first page and reports that more remain', async () => {
    mockListModels.mockResolvedValue(makePage([makeModel('a'), makeModel('b')], 1, 2));

    const { result } = renderSearch();

    await waitFor(() => expect(result.current.groups).toHaveLength(1));
    expect(result.current.groups[0].models.map((m) => m.name)).toEqual(['a', 'b']);
    expect(result.current.hasMore).toBe(true);
  });

  it('sends the search term as a case-insensitive substring filter', async () => {
    mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));
    const { result } = renderSearch();
    await waitFor(() => expect(mockListModels).toHaveBeenCalled());

    act(() => result.current.onSearchChange('  llama  '));

    await waitFor(() =>
      expect(mockListModels).toHaveBeenLastCalledWith(
        'ws1',
        expect.objectContaining({ filter: expect.objectContaining({ name: { $like: 'llama' } }) })
      )
    );
  });

  it('merges caller filters with the search term', async () => {
    mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

    renderSearch({ filter: { lora_enabled: true } });

    await waitFor(() =>
      expect(mockListModels).toHaveBeenCalledWith(
        'ws1',
        expect.objectContaining({ filter: { lora_enabled: true } })
      )
    );
  });

  it('keeps paging when include filters a whole page down to nothing', async () => {
    mockListModels
      .mockResolvedValueOnce(makePage([makeModel('no-provider')], 1, 2))
      .mockResolvedValueOnce(
        makePage([makeModel('served', { model_providers: ['ws1/build'] })], 2, 2)
      );

    const { result } = renderSearch({ include: (model) => !!model.model_providers?.length });

    await waitFor(() => expect(result.current.models.map((m) => m.name)).toEqual(['served']));
    expect(mockListModels).toHaveBeenCalledTimes(2);
    expect(result.current.hasMore).toBe(false);
  });

  it('appends the next page on demand', async () => {
    mockListModels
      .mockResolvedValueOnce(makePage([makeModel('a')], 1, 2))
      .mockResolvedValueOnce(makePage([makeModel('b')], 2, 2));

    const { result } = renderSearch();
    await waitFor(() => expect(result.current.hasMore).toBe(true));

    await act(async () => {
      await result.current.onLoadMore();
    });

    await waitFor(() => expect(result.current.models.map((m) => m.name)).toEqual(['a', 'b']));
    expect(result.current.hasMore).toBe(false);
  });

  describe('model filters', () => {
    const sized = (name: string, billions: number, family = 'llama') =>
      makeModel(name, { spec: { family, base_num_parameters: billions * 1_000_000_000 } as never });

    it('sends the provider and family to the server alongside the search term', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));
      const { result } = renderSearch({ filter: { lora_enabled: true } });
      await waitFor(() => expect(mockListModels).toHaveBeenCalled());

      act(() => result.current.modelFilters.onChange({ provider: 'ws1/build', family: 'llama' }));

      await waitFor(() =>
        expect(mockListModels).toHaveBeenLastCalledWith(
          'ws1',
          expect.objectContaining({
            filter: { lora_enabled: true, model_providers: 'ws1/build', family: 'llama' },
          })
        )
      );
    });

    it('filters by size in the browser without changing the request', async () => {
      mockListModels.mockResolvedValue(
        makePage([sized('tiny', 1), sized('mid', 7), sized('huge', 70)], 1, 1)
      );
      const { result } = renderSearch({ initialFilters: { size: 'medium' } });

      await waitFor(() => expect(result.current.models.map((m) => m.name)).toEqual(['mid']));
      expect(mockListModels).toHaveBeenCalledWith(
        'ws1',
        expect.not.objectContaining({ filter: expect.anything() })
      );
    });

    it('keeps paging when the size filter empties a whole page', async () => {
      mockListModels
        .mockResolvedValueOnce(makePage([sized('tiny', 1)], 1, 2))
        .mockResolvedValueOnce(makePage([sized('mid', 7)], 2, 2));

      const { result } = renderSearch({ initialFilters: { size: 'medium' } });

      await waitFor(() => expect(result.current.models.map((m) => m.name)).toEqual(['mid']));
    });

    it('lists providers as workspace-qualified options', async () => {
      mockProviders.data = [{ workspace: 'ws1', name: 'build' }];
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

      const { result } = renderSearch();

      expect(result.current.modelFilters.providerOptions).toEqual([
        { value: 'ws1/build', label: 'build' },
      ]);
    });

    it('offers a selected provider the list does not include', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

      const { result } = renderSearch({ initialFilters: { provider: 'other/gone' } });

      expect(result.current.modelFilters.providerOptions).toEqual([
        { value: 'other/gone', label: 'other/gone' },
      ]);
    });

    it('keeps every family it has seen as options after narrowing to one', async () => {
      mockListModels.mockResolvedValueOnce(
        makePage([sized('a', 1, 'llama'), sized('b', 1, 'mixtral')], 1, 1)
      );
      const { result } = renderSearch();
      await waitFor(() =>
        expect(result.current.modelFilters.familyOptions.map((o) => o.value)).toEqual([
          'llama',
          'mixtral',
        ])
      );

      mockListModels.mockResolvedValue(makePage([sized('a', 1, 'llama')], 1, 1));
      act(() => result.current.modelFilters.onChange({ family: 'llama' }));

      await waitFor(() => expect(result.current.models.map((m) => m.name)).toEqual(['a']));
      expect(result.current.modelFilters.familyOptions.map((o) => o.value)).toEqual([
        'llama',
        'mixtral',
      ]);
    });
  });
});
