// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFilesetsQueryKey, type ListFilesets } from '@nemo/common/src/api/filesets/filesetPages';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { isFilesetPurpose } from '@nemo/common/src/utils/filesetPurpose';
import type {
  FilesetOutput,
  FilesetPurpose,
  FilesListFilesetsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';

export const FILESETS_PAGE_SIZE = 20;

export interface UseFilesetSearchCoreOptions {
  workspace: string;
  listFilesets: ListFilesets;
  purpose?: FilesetPurpose | readonly FilesetPurpose[];
  description?: string;
  enabled?: boolean;
  pageSize?: number;
}

export interface UseFilesetSearchResult {
  filesets: FilesetOutput[];
  search: string;
  setSearch: (value: string) => void;
  loadMore: () => Promise<void>;
  hasMore: boolean;
  isLoading: boolean;
  isLoadingMore: boolean;
  isError: boolean;
  error: Error | null;
}

export const useFilesetSearchCore = ({
  workspace,
  listFilesets,
  purpose,
  description,
  enabled = true,
  pageSize = FILESETS_PAGE_SIZE,
}: UseFilesetSearchCoreOptions): UseFilesetSearchResult => {
  const [search, setSearch] = useState('');

  // Sorted so callers passing a fresh array literal each render share one cache entry.
  const purposeKey = typeof purpose === 'string' ? purpose : [...(purpose ?? [])].sort().join(',');

  const filter = useMemo<FilesListFilesetsParams['filter'] | undefined>(() => {
    const purposes = purposeKey.split(',').filter(isFilesetPurpose);
    const purposeClause =
      purposes.length === 1
        ? { purpose: purposes[0] }
        : purposes.length > 1
          ? { purpose: { $in: purposes } }
          : {};
    const clauses = {
      ...(search ? { name: { $like: `%${search}%` } } : {}),
      ...purposeClause,
      ...(description !== undefined ? { description: { $eq: description } } : {}),
    };
    return Object.keys(clauses).length
      ? withOperators<FilesListFilesetsParams['filter']>(clauses)
      : undefined;
  }, [search, purposeKey, description]);

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading, isError, error } =
    useInfiniteQuery({
      queryKey: [
        ...getFilesetsQueryKey(workspace),
        'infinite',
        'newest',
        purposeKey || 'all',
        description ?? null,
        search,
        pageSize,
      ] as const,
      queryFn: ({ signal, pageParam }) =>
        listFilesets(
          workspace,
          { page: pageParam, page_size: pageSize, sort: '-created_at', filter },
          signal
        ),
      initialPageParam: 1,
      getNextPageParam: (lastPage) => {
        const p = lastPage.pagination;
        return p && p.page < p.total_pages ? p.page + 1 : undefined;
      },
      enabled: enabled && !!workspace,
    });

  const filesets = useMemo(() => data?.pages.flatMap((page) => page.data) ?? [], [data?.pages]);

  const loadMore = useCallback(async () => {
    if (hasNextPage && !isFetchingNextPage) await fetchNextPage();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  return {
    filesets,
    search,
    setSearch,
    loadMore,
    hasMore: hasNextPage ?? false,
    isLoading,
    isLoadingMore: isFetchingNextPage,
    isError,
    error,
  };
};
