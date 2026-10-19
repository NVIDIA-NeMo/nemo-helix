// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { modelsListDeployments } from '@nemo/sdk/generated/platform/model-deployments';
import {
  ModelsListDeploymentsParams,
  ModelDeploymentsPage,
  ModelsListDeploymentsParams as ListModelDeploymentsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery, UseQueryOptions } from '@tanstack/react-query';
import { useEffect } from 'react';

import { DEFAULT_LARGE_PAGE_SIZE, QUERY_PREFIX_DEPLOYMENT_MANAGEMENT } from '../../constants/api';

export interface UseDmsModelsOptions {
  queryOptions?: Omit<UseQueryOptions<ModelDeploymentsPage, Error>, 'queryFn' | 'queryKey'>;
  query?: ModelsListDeploymentsParams;
  workspace: string;
}

/**
 * A basic filter for DMS models that fetches 1000 models sorted created at.
 */
export const BASIC_ALL_DMS_MODELS_DROPDOWN_FILTER: ListModelDeploymentsParams = {
  page_size: DEFAULT_LARGE_PAGE_SIZE,
  sort: 'created_at',
};

export const getQueryKeyInfiniteDmsModels = (query?: ModelsListDeploymentsParams) => [
  QUERY_PREFIX_DEPLOYMENT_MANAGEMENT,
  'models',
  'infinite',
  query,
];

/**
 * A wrapper for useInfiniteQuery that fetches DMS models.
 */
const useDmsModelsInfinite = (options?: UseDmsModelsOptions) => {
  return useInfiniteQuery({
    queryKey: getQueryKeyInfiniteDmsModels(options?.query),
    queryFn: ({ pageParam = 0 }) =>
      modelsListDeployments(options!.workspace, { ...options?.query, page: pageParam }),
    getNextPageParam: (lastPage: ModelDeploymentsPage) => {
      if (!lastPage.pagination) return undefined;
      const { page, total_pages } = lastPage.pagination;
      return page < total_pages ? page + 1 : undefined;
    },
    enabled: options?.queryOptions?.enabled !== false,
    initialPageParam: 1,
  });
};

/**
 * A hook for exhausting all DMS models.
 * This is useful for fetching all models from DMS.
 */
export const useAllDmsModels = (options?: UseDmsModelsOptions) => {
  const result = useDmsModelsInfinite(options);
  useEffect(() => {
    if (!result.isFetching && result.hasNextPage) {
      result.fetchNextPage();
    }
  }, [result]);
  return result;
};
