// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import {
  ModelsListModelsParams,
  ModelEntitysPage,
  ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery, UseQueryOptions } from '@tanstack/react-query';
import { useEffect } from 'react';

import { DEFAULT_LARGE_PAGE_SIZE, QUERY_PREFIX_PLATFORM } from '../../constants/api';

/**
 * A basic filter for a models dropdown that fetches models sorted alphabetically.
 * page_size is set high so most deployments are covered in a single page.
 */
export const BASIC_ALL_MODELS_DROPDOWN_FILTER: ModelsListModelsParams = {
  page_size: DEFAULT_LARGE_PAGE_SIZE,
  sort: 'name',
};

export interface UseModelsOptions {
  queryOptions?: Omit<UseQueryOptions<ModelEntitysPage, Error>, 'queryFn' | 'queryKey'>;
  query?: ModelsListModelsParams;
  workspace?: string;
}

export const getQueryKeyInfiniteModels = (workspace?: string, query?: ModelsListModelsParams) => [
  QUERY_PREFIX_PLATFORM,
  'models',
  'v2',
  'infinite',
  workspace,
  query,
];

/**
 * A wrapper for useInfiniteQuery that fetches models from the V2 platform API.
 *
 * Endpoint: GET /apis/models/v2/workspaces/{workspace}/models
 */
export const useModelsInfinite = (options?: UseModelsOptions) => {
  const workspace = options?.workspace ?? 'default';
  return useInfiniteQuery({
    queryKey: getQueryKeyInfiniteModels(workspace, options?.query),
    queryFn: ({ pageParam = 1 }) =>
      modelsListModels(workspace, { ...options?.query, page: pageParam }),
    getNextPageParam: (lastPage: ModelEntitysPage) => {
      if (!lastPage.pagination) return undefined;
      const { page, total_pages } = lastPage.pagination;
      return page < total_pages ? page + 1 : undefined;
    },
    enabled: options?.queryOptions?.enabled !== false,
    initialPageParam: 1,
  });
};

/**
 * Hook for exhaustively fetching all models from a single workspace.
 * Auto-fetches all pages until complete. Returns the raw infinite query result
 * (flat pages) suitable for table views with pagination.
 *
 * Endpoint: GET /apis/models/v2/workspaces/{workspace}/models
 */
export const useAllModels = (options?: UseModelsOptions) => {
  const result = useModelsInfinite(options);
  const { isFetching, hasNextPage, fetchNextPage } = result;
  useEffect(() => {
    if (!isFetching && hasNextPage) {
      fetchNextPage();
    }
  }, [isFetching, hasNextPage, fetchNextPage]);
  return result;
};

export type ModelWorkspaceGroup = {
  workspace: string;
  models: ModelEntity[];
};

export const buildWorkspaceGroup = (
  workspaceName: string,
  models: ModelEntity[]
): ModelWorkspaceGroup => ({
  workspace: workspaceName,
  models: [...models].sort((a, b) => (a.name ?? '').localeCompare(b.name ?? '')),
});
