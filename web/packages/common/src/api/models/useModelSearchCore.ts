// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import {
  EMPTY_MODEL_FILTERS,
  MODEL_CAPABILITY_OPTIONS,
  MODEL_INPUT_MODALITY_OPTIONS,
  MODEL_SIZE_OPTIONS,
  modelSizeFilter,
  withSelectedOptions,
  type ModelFilterControls,
  type ModelFilterOption,
  type ModelFilterValues,
  type ParameterCountFilter,
} from '@nemo/common/src/api/models/modelFilters';
import {
  getNextModelsPageParam,
  getQueryKeyInfiniteModels,
  getQueryKeyModelProviderOptions,
  type ListModels,
  type ListProviders,
  type ModelWorkspaceGroup,
} from '@nemo/common/src/api/models/modelPages';
import { groupModelsByWorkspace } from '@nemo/common/src/utils/models';
import type { ModelEntity, ModelEntityFilter } from '@nemo/sdk/generated/platform/schema';
import { ModelEntitySortField } from '@nemo/sdk/generated/platform/schema/ModelEntitySortField';
import { skipToken, useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useState } from 'react';

/**
 * Page size for search-as-you-type model lists. Small on purpose: the dropdown pulls the next
 * page as the user scrolls, so the first page needs to arrive fast, not be complete.
 */
export const MODEL_SEARCH_PAGE_SIZE = 25;

export type ModelSearchFilter = WithFilterOperators<ModelEntityFilter>;

// The API JSON-decodes operands that start with a bracket; a leading % is a no-op in its %term% match.
const toLikeOperand = (term: string): string => {
  if (term.startsWith('[') || term.startsWith('{')) {
    return `%${term}`;
  }
  return term;
};

const providerFilter = (providers: string[]): ModelSearchFilter['model_providers'] => {
  if (providers.length === 1) {
    return providers[0];
  }
  return { $in: providers };
};

export interface UseModelSearchCoreOptions {
  /** Workspace to search. The query stays idle while this is null. */
  workspace: string | null;
  listModels: ListModels;
  /** Fills the provider filter's options. Without it the provider filter offers only its current value. */
  listProviders?: ListProviders;
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
  initialFilters?: ModelFilterValues;
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
  modelFilters: ModelFilterControls;
  error: Error | null;
}

export interface UseModelSearchResult extends ModelSearchProps {
  models: ModelEntity[];
  search: string;
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
  listProviders,
  filter,
  sort = ModelEntitySortField.name,
  pageSize = MODEL_SEARCH_PAGE_SIZE,
  enabled = true,
  include,
  initialFilters = EMPTY_MODEL_FILTERS,
}: UseModelSearchCoreOptions): UseModelSearchResult => {
  const [search, setSearch] = useState('');
  const [filterState, setFilterState] = useState({ workspace, values: initialFilters });
  const filters = filterState.workspace === workspace ? filterState.values : initialFilters;
  const setFilters = useCallback(
    (values: ModelFilterValues) => setFilterState({ workspace, values }),
    [workspace]
  );

  const query = useMemo(() => {
    const trimmed = search.trim();
    const { providers, capabilities, inputModalities, sizes } = filters;
    const parameterCount = sizes?.length ? modelSizeFilter(sizes) : undefined;
    const merged: Omit<ModelSearchFilter, 'parameter_count'> & {
      parameter_count?: ModelSearchFilter['parameter_count'] | ParameterCountFilter;
    } = {
      ...filter,
      ...(trimmed ? { name: { $like: toLikeOperand(trimmed) } } : {}),
      ...(providers?.length ? { model_providers: providerFilter(providers) } : {}),
      ...(capabilities?.length ? { capabilities: { $in: capabilities } } : {}),
      ...(inputModalities?.length ? { input_modalities: { $in: inputModalities } } : {}),
      ...(parameterCount ? { parameter_count: parameterCount } : {}),
    };
    return {
      page_size: pageSize,
      sort,
      ...(Object.keys(merged).length > 0 ? { filter: merged as ModelEntityFilter } : {}),
    };
  }, [filter, filters, pageSize, search, sort]);

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
    isPending,
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

  const { data: providers, isLoading: providersLoading } = useQuery({
    queryKey: getQueryKeyModelProviderOptions(queryWorkspace),
    queryFn: listProviders
      ? () =>
          fetchAllPages((page, page_size) =>
            listProviders(queryWorkspace, { page, page_size, sort: 'name' })
          )
      : skipToken,
    enabled: isEnabled,
  });

  const providerOptions = useMemo<ModelFilterOption[]>(
    () =>
      withSelectedOptions(
        (providers ?? []).map((provider) => ({
          value: `${provider.workspace}/${provider.name}`,
          label: provider.name,
        })),
        filters.providers
      ),
    [filters.providers, providers]
  );

  const modelFilters = useMemo<ModelFilterControls>(
    () => ({
      values: filters,
      onChange: setFilters,
      providerOptions,
      capabilityOptions: MODEL_CAPABILITY_OPTIONS,
      inputModalityOptions: MODEL_INPUT_MODALITY_OPTIONS,
      sizeOptions: MODEL_SIZE_OPTIONS,
      providersLoading: !!listProviders && providersLoading,
    }),
    [filters, listProviders, providerOptions, providersLoading, setFilters]
  );

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
    // An enabled initial query remains pending when offline, even while it is not fetching.
    loading: isEnabled && isPending,
    onSearchChange: setSearch,
    onLoadMore,
    hasMore,
    isLoadingMore: isFetchingNextPage,
    modelFilters,
  };
};
