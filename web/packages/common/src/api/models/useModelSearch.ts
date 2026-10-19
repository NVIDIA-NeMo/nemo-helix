// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import {
  collectModelFamilies,
  EMPTY_MODEL_FILTERS,
  matchesModelSize,
  MODEL_SIZE_OPTIONS,
  withSelectedOption,
  type ModelFilterControls,
  type ModelFilterOption,
  type ModelFilterValues,
} from '@nemo/common/src/api/models/modelFilters';
import { useModelsInfinite, type ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import { groupModelsByWorkspace } from '@nemo/common/src/utils/models';
import { useModelsListProviders } from '@nemo/sdk/generated/platform/model-providers';
import {
  type ModelEntity,
  ModelEntitySortField,
  type ModelEntityFilter,
} from '@nemo/sdk/generated/platform/schema';
import { useCallback, useEffect, useMemo, useState } from 'react';

const PROVIDER_OPTIONS_PAGE_SIZE = 100;

/**
 * Page size for search-as-you-type model lists. Small on purpose: the dropdown pulls the next
 * page as the user scrolls, so the first page needs to arrive fast, not be complete.
 */
export const MODEL_SEARCH_PAGE_SIZE = 25;

export type ModelSearchFilter = WithFilterOperators<ModelEntityFilter>;

export interface UseModelSearchOptions {
  /** Workspace to search. The query stays idle while this is null. */
  workspace: string | null;
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
 * Props for `ModelSelectV2`, ready to spread. Every field lines up with a prop name so a caller
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
 * @example
 * const [open, setOpen] = useState(false);
 * const models = useModelSearch({ workspace, enabled: open });
 * return <ModelSelectV2 {...models} value={value} onValueChange={onChange} onOpenChange={setOpen} />;
 */
export const useModelSearch = ({
  workspace,
  filter,
  sort = ModelEntitySortField.name,
  pageSize = MODEL_SEARCH_PAGE_SIZE,
  enabled = true,
  include,
  initialFilters = EMPTY_MODEL_FILTERS,
}: UseModelSearchOptions): UseModelSearchResult => {
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState<ModelFilterValues>(initialFilters);
  const [knownFamilies, setKnownFamilies] = useState<string[]>([]);

  const query = useMemo(() => {
    const trimmed = search.trim();
    const merged: ModelSearchFilter = {
      ...filter,
      ...(trimmed ? { name: { $like: trimmed } } : {}),
      ...(filters.provider ? { model_providers: filters.provider } : {}),
      ...(filters.family ? { family: filters.family } : {}),
    };
    return {
      page_size: pageSize,
      sort,
      ...(Object.keys(merged).length > 0 ? { filter: merged as ModelEntityFilter } : {}),
    };
  }, [filter, filters.family, filters.provider, pageSize, search, sort]);

  const isEnabled = enabled && !!workspace;
  const { data, error, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading } =
    useModelsInfinite({
      workspace: workspace ?? undefined,
      query,
      queryOptions: { enabled: isEnabled },
    });

  const loadedModels = useMemo(
    () => data?.pages.flatMap((page) => page.data ?? []) ?? [],
    [data?.pages]
  );

  const size = filters.size;
  const includeModel = useMemo(() => {
    if (!size && !include) return undefined;
    return (model: ModelEntity) =>
      (!size || matchesModelSize(model, size)) && (!include || include(model));
  }, [include, size]);

  const models = useMemo(
    () => (includeModel ? loadedModels.filter(includeModel) : loadedModels),
    [includeModel, loadedModels]
  );

  const groups = useMemo(() => groupModelsByWorkspace(models, { sort: true }), [models]);

  const { data: providersPage, isLoading: providersLoading } = useModelsListProviders(
    workspace ?? '',
    { page_size: PROVIDER_OPTIONS_PAGE_SIZE, sort: 'name' },
    { query: { enabled: isEnabled } }
  );

  useEffect(() => {
    const families = collectModelFamilies(loadedModels).map((option) => option.value);
    if (families.length === 0) return;
    setKnownFamilies((known) => {
      const merged = new Set([...known, ...families]);
      return merged.size === known.length ? known : [...merged];
    });
  }, [loadedModels]);

  const providerOptions = useMemo<ModelFilterOption[]>(
    () =>
      withSelectedOption(
        (providersPage?.data ?? []).map((provider) => ({
          value: `${provider.workspace}/${provider.name}`,
          label: provider.name,
        })),
        filters.provider
      ),
    [filters.provider, providersPage?.data]
  );

  const familyOptions = useMemo(
    () =>
      withSelectedOption(
        [...knownFamilies]
          .sort((a, b) => a.localeCompare(b))
          .map((family) => ({ value: family, label: family })),
        filters.family
      ),
    [filters.family, knownFamilies]
  );

  const modelFilters = useMemo<ModelFilterControls>(
    () => ({
      values: filters,
      onChange: setFilters,
      providerOptions,
      familyOptions,
      sizeOptions: MODEL_SIZE_OPTIONS,
      providersLoading,
    }),
    [familyOptions, filters, providerOptions, providersLoading]
  );

  const hasMore = !!hasNextPage;

  const onLoadMore = useCallback(async () => {
    if (!hasNextPage || isFetchingNextPage) return;
    await fetchNextPage();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  // `include` can empty a whole page, leaving the list with no rows to scroll and therefore no way
  // to ask for the next one. Keep paging until something survives the filter.
  useEffect(() => {
    if (isEnabled && models.length === 0 && hasNextPage && !isFetchingNextPage && !isLoading) {
      void fetchNextPage();
    }
  }, [fetchNextPage, hasNextPage, isEnabled, isFetchingNextPage, isLoading, models.length]);

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
    modelFilters,
  };
};
