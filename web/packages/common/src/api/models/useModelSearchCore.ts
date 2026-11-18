// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import {
  getNextModelsPageParam,
  getQueryKeyInfiniteModels,
  type ListModels,
  type ModelWorkspaceGroup,
} from '@nemo/common/src/api/models/modelPages';
import { groupModelsByWorkspace } from '@nemo/common/src/utils/models';
import type { ModelEntity, ModelEntityFilter } from '@nemo/sdk/generated/platform/schema';
import { ModelEntitySortField } from '@nemo/sdk/generated/platform/schema/ModelEntitySortField';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useState } from 'react';

/**
 * Page size for search-as-you-type model lists. Small on purpose: the dropdown pulls the next
 * page as the user scrolls, so the first page needs to arrive fast, not be complete.
 */
export const MODEL_SEARCH_PAGE_SIZE = 25;

export type ModelSearchFilter = WithFilterOperators<ModelEntityFilter>;

export interface UseModelSearchCoreOptions {
  /** Workspace to search. The query stays idle while this is null. */
  workspace: string | null;
  listModels: ListModels;
  /** Extra filters merged into the request (e.g. `lora_enabled`, `base_model`). */
  filter?: ModelSearchFilter;
  sort?: ModelEntitySortField;
  pageSize?: number;
  enabled?: boolean;
  /**
   * Client-side predicate applied to every page — for conditions the API cannot express, such as
   * "has a ready deployment" (`model_providers.length > 0`). The hook keeps paging while a page
   * filters down to nothing, so an excluded page never stalls the list.
   */
  include?: (model: ModelEntity) => boolean;
}

/**
 * Props for `ModelSelect`, ready to spread. Every field lines up with a prop name so a caller
 * that needs nothing custom is a single line.
 */
export interface ModelSearchProps {
  groups: ModelWorkspaceGroup[];
  loading: boolean;
  onSearchChange: (search: string) => void;
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  isLoadingMore: boolean;
}

export interface UseModelSearchResult extends ModelSearchProps {
  models: ModelEntity[];
  search: string;
  error: Error | null;
}

/**
 * Server-side model search with progressive paging — the counterpart to `useAllModels`, which
 * walks every page up front. Filtering happens in the API and pages arrive as the user scrolls,
 * so a workspace with thousands of models costs one small request at a time.
 *
 * Takes its fetcher so plugins can run it on `host.sdk`; Studio code uses `useModelSearch`.
 *
 * @example
 * const [open, setOpen] = useState(false);
 * const models = useModelSearchCore({ workspace, listModels, enabled: open });
 * return <ModelSelect {...models} value={value} onValueChange={onChange} onOpenChange={setOpen} />;
 */
export const useModelSearchCore = ({
  workspace,
  listModels,
  filter,
  sort = ModelEntitySortField.name,
  pageSize = MODEL_SEARCH_PAGE_SIZE,
  enabled = true,
  include,
}: UseModelSearchCoreOptions): UseModelSearchResult => {
  const [search, setSearch] = useState('');

  const query = useMemo(() => {
    const trimmed = search.trim();
    const merged: ModelSearchFilter = {
      ...filter,
      ...(trimmed ? { name: { $like: trimmed } } : {}),
    };
    return {
      page_size: pageSize,
      sort,
      ...(Object.keys(merged).length > 0 ? { filter: merged as ModelEntityFilter } : {}),
    };
  }, [filter, pageSize, search, sort]);

  const isEnabled = enabled && !!workspace;
  const queryWorkspace = workspace ?? 'default';
  const {
    data,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchNextPageError,
    isFetchingNextPage,
    isLoading,
  } = useInfiniteQuery({
    queryKey: getQueryKeyInfiniteModels(queryWorkspace, query),
    queryFn: ({ pageParam }) => listModels(queryWorkspace, { ...query, page: pageParam }),
    getNextPageParam: getNextModelsPageParam,
    enabled: isEnabled,
    initialPageParam: 1,
  });

  const models = useMemo(() => {
    const loaded = data?.pages.flatMap((page) => page.data ?? []) ?? [];
    return include ? loaded.filter(include) : loaded;
  }, [data?.pages, include]);

  const groups = useMemo(() => groupModelsByWorkspace(models, { sort: true }), [models]);

  const hasMore = !!hasNextPage;

  const onLoadMore = useCallback(async () => {
    if (!hasNextPage || isFetchingNextPage) return;
    await fetchNextPage();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  // `include` can empty a whole page, leaving the list with no rows to scroll and therefore no way
  // to ask for the next one. Keep paging until something survives the filter.
  useEffect(() => {
    if (
      isEnabled &&
      models.length === 0 &&
      hasNextPage &&
      !isFetchingNextPage &&
      !isFetchNextPageError &&
      !isLoading
    ) {
      void fetchNextPage();
    }
  }, [
    fetchNextPage,
    hasNextPage,
    isEnabled,
    isFetchNextPageError,
    isFetchingNextPage,
    isLoading,
    models.length,
  ]);

  return {
    models,
    groups,
    search,
    error,
    loading: isLoading,
    onSearchChange: setSearch,
    onLoadMore,
    hasMore,
    isLoadingMore: isFetchingNextPage,
  };
};
