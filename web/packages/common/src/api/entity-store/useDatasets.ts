// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesListFilesets } from '@nemo/sdk/generated/platform/files';
import type {
  FilesetOutputsPage,
  FilesListFilesetsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery, UseQueryOptions } from '@tanstack/react-query';

export const DEFAULT_FILESETS_PAGE_SIZE = 25;

/**
 * @deprecated Use DEFAULT_FILESETS_PAGE_SIZE instead
 */
export const DEFAULT_DATASETS_PAGE_SIZE = DEFAULT_FILESETS_PAGE_SIZE;

export interface UseFilesetsInfiniteOptions {
  workspace: string;
  queryOptions?: Omit<UseQueryOptions<FilesetOutputsPage, Error>, 'queryFn' | 'queryKey'>;
  query?: Omit<FilesListFilesetsParams, 'page' | 'page_size'>;
}

/**
 * @deprecated Use UseFilesetsInfiniteOptions instead
 */
export type UseDatasetsInfiniteOptions = UseFilesetsInfiniteOptions;

export const getQueryKeyInfiniteFilesets = (
  workspace: string,
  query?: Omit<FilesListFilesetsParams, 'page' | 'page_size'>
) => ['filesets', 'infinite', workspace, query];

/**
 * @deprecated Use getQueryKeyInfiniteFilesets instead
 */
export const getQueryKeyInfiniteDatasets = getQueryKeyInfiniteFilesets;

/**
 * A wrapper for useInfiniteQuery that fetches filesets with pagination.
 * Supports search via the `filter` parameter.
 */
export const useFilesetsInfinite = (options: UseFilesetsInfiniteOptions) => {
  const { workspace, query, queryOptions } = options;

  return useInfiniteQuery({
    queryKey: getQueryKeyInfiniteFilesets(workspace, query),
    queryFn: async ({ pageParam = 1 }) => {
      return filesListFilesets(workspace, {
        page_size: DEFAULT_FILESETS_PAGE_SIZE,
        ...query,
        page: pageParam,
      });
    },
    getNextPageParam: (lastPage: FilesetOutputsPage) => {
      if (!lastPage.pagination) return undefined;
      const { page, total_pages } = lastPage.pagination;
      return page < total_pages ? page + 1 : undefined;
    },
    enabled: queryOptions?.enabled !== false && Boolean(workspace),
    initialPageParam: 1,
  });
};

/**
 * @deprecated Use useFilesetsInfinite instead
 */
export const useDatasetsInfinite = useFilesetsInfinite;
