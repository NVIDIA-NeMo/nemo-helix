// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ListModels, ListProviders } from '@nemo/common/src/api/models/modelPages';
import { useModelSearchCore } from '@nemo/common/src/api/models/useModelSearchCore';
import type {
  ModelEntity,
  ModelEntitysPage,
  ModelProvidersPage,
} from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';

const mockListModels = vi.fn<ListModels>();
const mockListProviders = vi.fn<ListProviders>();

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

const renderSearch = (options: Partial<Parameters<typeof useModelSearchCore>[0]> = {}) =>
  renderHook(
    () =>
      useModelSearchCore({
        workspace: 'ws1',
        listModels: mockListModels,
        listProviders: mockListProviders,
        ...options,
      }),
    {
      wrapper: createWrapper(),
    }
  );

beforeEach(() => {
  mockListModels.mockReset();
  mockListProviders.mockReset();
  mockListProviders.mockResolvedValue({ data: [] } as unknown as ModelProvidersPage);
});

describe('useModelSearchCore', () => {
  it('stays idle while disabled', () => {
    renderSearch({ enabled: false });
    expect(mockListModels).not.toHaveBeenCalled();
  });

  it('stays idle without a workspace', () => {
    renderHook(() => useModelSearchCore({ workspace: null, listModels: mockListModels }), {
      wrapper: createWrapper(),
    });
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

  it('stops auto-paging after a next page fails', async () => {
    mockListModels
      .mockResolvedValueOnce(makePage([makeModel('no-provider')], 1, 3))
      .mockImplementation(
        () => new Promise((_, reject) => setTimeout(() => reject(new Error('boom')), 10))
      );

    const { result } = renderSearch({ include: (model) => !!model.model_providers?.length });

    await waitFor(() => expect(result.current.error).toBeTruthy());
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(mockListModels).toHaveBeenCalledTimes(2);
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
    it('sends each filter as an any-of list alongside caller filters', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));
      const { result } = renderSearch({ filter: { lora_enabled: true } });
      await waitFor(() => expect(mockListModels).toHaveBeenCalled());

      act(() =>
        result.current.modelFilters.onChange({
          providers: ['ws1/build', 'ws1/local'],
          capabilities: ['tool_calling', 'reasoning'],
          inputModalities: ['image'],
        })
      );

      await waitFor(() =>
        expect(mockListModels).toHaveBeenLastCalledWith(
          'ws1',
          expect.objectContaining({
            filter: {
              lora_enabled: true,
              model_providers: { $in: ['ws1/build', 'ws1/local'] },
              capabilities: { $in: ['tool_calling', 'reasoning'] },
              input_modalities: { $in: ['image'] },
            },
          })
        )
      );
    });

    it('sends the size as a parameter count range', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('mid')], 1, 1));

      renderSearch({ initialFilters: { sizes: ['medium'] } });

      await waitFor(() =>
        expect(mockListModels).toHaveBeenCalledWith(
          'ws1',
          expect.objectContaining({
            filter: { parameter_count: { $gte: 3_000_000_000, $lt: 10_000_000_000 } },
          })
        )
      );
    });

    it('leaves out filters whose lists are empty', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

      renderSearch({ initialFilters: { providers: [], capabilities: [], sizes: [] } });

      await waitFor(() =>
        expect(mockListModels).toHaveBeenCalledWith(
          'ws1',
          expect.not.objectContaining({ filter: expect.anything() })
        )
      );
    });

    it('lists providers as workspace-qualified options', async () => {
      mockListProviders.mockResolvedValue({
        data: [{ workspace: 'default', name: 'nvidia-build' }],
      } as unknown as ModelProvidersPage);
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

      const { result } = renderSearch();

      await waitFor(() =>
        expect(result.current.modelFilters.providerOptions).toEqual([
          { value: 'default/nvidia-build', label: 'nvidia-build' },
        ])
      );
    });

    it('offers a selected provider the list does not include', async () => {
      mockListModels.mockResolvedValue(makePage([makeModel('a')], 1, 1));

      const { result } = renderSearch({ initialFilters: { providers: ['other/gone'] } });

      expect(result.current.modelFilters.providerOptions).toEqual([
        { value: 'other/gone', label: 'other/gone' },
      ]);
    });
  });
});
