// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import { ModelsListModelsParams, ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery, UseQueryOptions } from '@tanstack/react-query';
import { useEffect } from 'react';

import { DEFAULT_LARGE_PAGE_SIZE, QUERY_PREFIX_ENTITY_STORE } from '../../constants/api';

export interface UseModelsOptions {
  queryOptions?: Omit<UseQueryOptions<ModelEntitysPage, Error>, 'queryFn' | 'queryKey'>;
  query?: ModelsListModelsParams;
  workspace?: string;
}

/**
 * A basic filter for a models dropdown that fetches 1000 models sorted alphabetically.
 */
export const BASIC_ALL_MODELS_DROPDOWN_FILTER: ModelsListModelsParams = {
  page_size: DEFAULT_LARGE_PAGE_SIZE,
  sort: 'name',
};

export const getQueryKeyInfiniteModels = (workspace?: string, query?: ModelsListModelsParams) => [
  QUERY_PREFIX_ENTITY_STORE,
  'models',
  'infinite',
  workspace,
  query,
];

/**
 * A wrapper for useInfiniteQuery that fetches entity store models.
 */
const useModelsInfinite = (options?: UseModelsOptions) => {
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
 * A hook for exhausting all entity store models.
 * This is useful for fetching all models for the model dropdown.
 */
export const useAllModels = (options?: UseModelsOptions) => {
  const result = useModelsInfinite(options);
  useEffect(() => {
    if (!result.isFetching && result.hasNextPage) {
      result.fetchNextPage();
    }
  }, [result]);
  return result;
};
