// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { ResourceRef } from '@nemo/common/src/types';
import {
  filterModel,
  getBaseModelURN,
  getModelEntityChatStatus,
} from '@nemo/common/src/utils/models';
import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import type {
  ModelEntitysPage,
  ModelEntity,
  ModelsListModelsParams,
} from '@nemo/sdk/generated/platform/schema';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useEffect, useMemo } from 'react';

const MODEL_FETCH_LIMIT = 1000;

/**
 * Hook for fetching judge models from the current workspace.
 * This is used for the LLM-as-a-Judge metric configuration.
 */
export const useJudgeModels = ({ enabled, search }: { enabled?: boolean; search?: string }) => {
  const workspace = useWorkspaceFromPath();

  const { data, isFetching, isLoading, error, hasNextPage, fetchNextPage } = useInfiniteQuery({
    queryKey: ['judge-models', workspace, search],
    queryFn: ({ pageParam = 1 }) =>
      modelsListModels('-', {
        page: pageParam,
        page_size: MODEL_FETCH_LIMIT,
        sort: '-created_at',
        filter: JSON.stringify({
          workspace: { $eq: workspace },
          ...(search ? { name: { $like: search } } : {}),
        }) as unknown as ModelsListModelsParams['filter'],
      }),
    getNextPageParam: (lastPage: ModelEntitysPage) => {
      if (!lastPage.pagination) return undefined;
      const { page, total_pages } = lastPage.pagination;
      return page < total_pages ? page + 1 : undefined;
    },
    enabled,
    initialPageParam: 1,
  });

  // Auto-fetch all pages
  useEffect(() => {
    if (!isFetching && hasNextPage) {
      fetchNextPage();
    }
  }, [isFetching, hasNextPage, fetchNextPage]);

  const allData = useMemo(() => {
    const allModels = data?.pages.flatMap((page) => page.data) ?? [];
    const uniqueModels = allModels.reduce<Record<ResourceRef, ModelEntity>>((acc, model) => {
      const urn = getURNFromNamedEntityRef(model);
      if (urn && !acc[urn]) {
        acc[urn] = model;
      }
      return acc;
    }, {});

    return Object.values(uniqueModels)
      .filter((model) => filterModel(model, search))
      .filter((model) => {
        const modelURN = getURNFromNamedEntityRef(model);
        const baseModelURN = getBaseModelURN(model) as ResourceRef;
        const chatStatus = getModelEntityChatStatus(model);

        // Must have a URN and be a chat model (not disabled)
        if ((!baseModelURN && !modelURN) || chatStatus === 'disabled') {
          return false;
        }

        // Show all chat-capable models
        return true;
      });
  }, [data?.pages, search]);

  return {
    data: allData,
    isLoading: isLoading || isFetching,
    error: error,
  };
};
