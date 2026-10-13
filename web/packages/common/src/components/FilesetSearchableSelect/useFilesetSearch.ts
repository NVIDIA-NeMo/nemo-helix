// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import {
  filesListFilesets,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import type {
  FilesetOutput,
  FilesetPurpose,
  FilesListFilesetsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';

/** The v2 API caps `page_size` at 100; 20 keeps the first paint small and pages on scroll. */
export const FILESETS_PAGE_SIZE = 20;

export interface UseFilesetSearchOptions {
  workspace: string;
  /** Restrict to one fileset `purpose`. Omit to list every purpose. */
  purpose?: FilesetPurpose;
  /** Restrict to filesets carrying this exact `description`, which is how Studio
   *  tags filesets it creates for a specific job. Omit to list every fileset. */
  description?: string;
  enabled?: boolean;
  pageSize?: number;
}

export interface UseFilesetSearchResult {
  /** Every fileset loaded so far, newest first. */
  filesets: FilesetOutput[];
  search: string;
  setSearch: (value: string) => void;
  loadMore: () => Promise<void>;
  hasMore: boolean;
  isLoading: boolean;
  isLoadingMore: boolean;
  /** The fileset query failed; the caller should say so rather than show an empty list. */
  isError: boolean;
  error: Error | null;
}

/**
 * Search + paginate a workspace's filesets.
 *
 * Server-side on both counts: name search goes out as a `$like` filter and results are
 * paged, so this does not silently truncate the way a single capped page does. Sorted
 * newest-first, since a fileset the user just created is the one they are looking for.
 */
export const useFilesetSearch = ({
  workspace,
  purpose,
  description,
  enabled = true,
  pageSize = FILESETS_PAGE_SIZE,
}: UseFilesetSearchOptions): UseFilesetSearchResult => {
  const [search, setSearch] = useState('');

  const filter = useMemo<FilesListFilesetsParams['filter'] | undefined>(() => {
    const clauses = {
      ...(search ? { name: { $like: `%${search}%` } } : {}),
      ...(purpose ? { purpose } : {}),
      ...(description ? { description: { $eq: description } } : {}),
    };
    return Object.keys(clauses).length
      ? withOperators<FilesListFilesetsParams['filter']>(clauses)
      : undefined;
  }, [search, purpose, description]);

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading, isError, error } =
    useInfiniteQuery({
      queryKey: [
        ...getFilesListFilesetsQueryKey(workspace),
        'infinite',
        'newest',
        purpose ?? 'all',
        description ?? 'any',
        search,
        pageSize,
      ] as const,
      queryFn: ({ signal, pageParam }) =>
        filesListFilesets(
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
