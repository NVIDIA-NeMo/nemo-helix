// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildWorkspaceGroup,
  type ModelWorkspaceGroup,
  useModelsInfinite,
} from '@nemo/common/src/api/models/useModels';
import type { ModelsListModelsParams, ModelEntitysPage } from '@nemo/sdk/generated/platform/schema';
import type { UseQueryOptions } from '@tanstack/react-query';
import { useEffect, useMemo } from 'react';

export interface UseModelsFromWorkspaceOptions {
  queryOptions?: Omit<UseQueryOptions<ModelEntitysPage, Error>, 'queryFn' | 'queryKey'>;
  query?: ModelsListModelsParams;
  workspace: string | null;
}

/**
 * Fetches models from a single workspace (no shared/default workspace).
 *
 * Single-workspace strategy for model dropdowns:
 * - Uses the caller-provided query with use-case-specific filter
 * - Filters for available models (model_providers.length > 0 → deployment reached READY)
 * - Returns grouped structure: workspace → models (for dropdowns)
 *
 * Endpoint: GET /apis/models/v2/workspaces/{workspace}/models
 */
export const useModelsFromWorkspace = (options: UseModelsFromWorkspaceOptions) => {
  const { workspace, query, queryOptions } = options;

  const currentWorkspaceResult = useModelsInfinite({
    workspace: workspace ?? undefined,
    query,
    queryOptions: {
      ...queryOptions,
      enabled: queryOptions?.enabled !== false && !!workspace,
    },
  });

  const {
    isFetching: isCurrentFetching,
    hasNextPage: currentHasNext,
    fetchNextPage: fetchCurrentNext,
  } = currentWorkspaceResult;
  useEffect(() => {
    if (!isCurrentFetching && currentHasNext) {
      fetchCurrentNext();
    }
  }, [isCurrentFetching, currentHasNext, fetchCurrentNext]);

  const groups = useMemo((): ModelWorkspaceGroup[] => {
    const models =
      currentWorkspaceResult.data?.pages.flatMap((page) =>
        Array.isArray(page.data) ? page.data : []
      ) ?? [];

    if (!workspace || models.length === 0) return [];
    return [buildWorkspaceGroup(workspace, models)];
  }, [currentWorkspaceResult.data, workspace]);

  return {
    groups,
    isFetching: currentWorkspaceResult.isFetching || currentWorkspaceResult.hasNextPage,
    isError: currentWorkspaceResult.isError,
    error: currentWorkspaceResult.error,
  };
};
