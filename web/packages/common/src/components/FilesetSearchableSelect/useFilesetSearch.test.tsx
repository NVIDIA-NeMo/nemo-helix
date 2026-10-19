// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { useFilesetSearch } from '@nemo/common/src/components/FilesetSearchableSelect/useFilesetSearch';
import { filesListFilesets } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesListFilesets: vi.fn(),
  getFilesListFilesetsQueryKey: vi.fn((workspace: string) => ['filesets', workspace]),
}));

const fileset = (name: string) => ({ id: `default/${name}`, name, workspace: 'default' });

const page = (names: string[], pageNumber: number, totalPages: number) =>
  ({
    data: names.map(fileset) as FilesetOutput[],
    pagination: { page: pageNumber, total_pages: totalPages },
  }) as Awaited<ReturnType<typeof filesListFilesets>>;

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
};

/** A wrapper backed by one client, so two hook instances share a cache. */
const createSharedWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useFilesetSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('requests the first page newest-first, at the shared page size', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });

    await waitFor(() => expect(result.current.filesets).toHaveLength(1));
    expect(filesListFilesets).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ page: 1, page_size: 20, sort: '-created_at' }),
      expect.anything()
    );
  });

  it('accumulates pages instead of truncating at the first one', async () => {
    vi.mocked(filesListFilesets)
      .mockResolvedValueOnce(page(['a', 'b'], 1, 2))
      .mockResolvedValueOnce(page(['c'], 2, 2));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });

    await waitFor(() => expect(result.current.filesets).toHaveLength(2));
    expect(result.current.hasMore).toBe(true);

    await act(async () => {
      await result.current.loadMore();
    });

    await waitFor(() => expect(result.current.filesets).toHaveLength(3));
    expect(result.current.filesets.map((f) => f.name)).toEqual(['a', 'b', 'c']);
    expect(result.current.hasMore).toBe(false);
  });

  it('stops paging at the last page', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });

    await waitFor(() => expect(result.current.hasMore).toBe(false));
    await act(async () => {
      await result.current.loadMore();
    });
    expect(filesListFilesets).toHaveBeenCalledTimes(1);
  });

  it('sends the search term as a server-side $like filter', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });
    await waitFor(() => expect(result.current.filesets).toHaveLength(1));

    act(() => result.current.setSearch('pay'));

    await waitFor(() =>
      expect(filesListFilesets).toHaveBeenCalledWith(
        'ws',
        expect.objectContaining({ filter: withOperators({ name: { $like: '%pay%' } }) }),
        expect.anything()
      )
    );
  });

  it('combines search and purpose into one filter', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws', purpose: 'generic' }), {
      wrapper,
    });
    await waitFor(() => expect(result.current.filesets).toHaveLength(1));

    act(() => result.current.setSearch('pay'));

    await waitFor(() =>
      expect(filesListFilesets).toHaveBeenCalledWith(
        'ws',
        expect.objectContaining({
          filter: withOperators({ name: { $like: '%pay%' }, purpose: 'generic' }),
        }),
        expect.anything()
      )
    );
  });

  it('sends no filter when unfiltered, so every purpose is listed', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });

    await waitFor(() => expect(result.current.filesets).toHaveLength(1));
    expect(filesListFilesets).toHaveBeenCalledWith(
      'ws',
      expect.objectContaining({ filter: undefined }),
      expect.anything()
    );
  });

  it('caches per page size, so a differently-sized hook does not reuse the wrong page', async () => {
    vi.mocked(filesListFilesets).mockResolvedValue(page(['a'], 1, 1));
    const sharedWrapper = createSharedWrapper();

    const { result: defaultSized } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), {
      wrapper: sharedWrapper,
    });
    await waitFor(() => expect(defaultSized.current.filesets).toHaveLength(1));

    const { result: smallPaged } = renderHook(
      () => useFilesetSearch({ workspace: 'ws', pageSize: 5 }),
      { wrapper: sharedWrapper }
    );
    await waitFor(() => expect(smallPaged.current.filesets).toHaveLength(1));

    expect(filesListFilesets).toHaveBeenCalledTimes(2);
    expect(filesListFilesets).toHaveBeenLastCalledWith(
      'ws',
      expect.objectContaining({ page_size: 5 }),
      expect.anything()
    );
  });

  it('reports the failure instead of an empty result set', async () => {
    vi.mocked(filesListFilesets).mockRejectedValue(new Error('boom'));

    const { result } = renderHook(() => useFilesetSearch({ workspace: 'ws' }), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error?.message).toBe('boom');
    expect(result.current.filesets).toEqual([]);
  });

  it('does not query without a workspace', () => {
    renderHook(() => useFilesetSearch({ workspace: '' }), { wrapper });
    expect(filesListFilesets).not.toHaveBeenCalled();
  });
});
