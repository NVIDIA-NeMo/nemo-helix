// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import { modelsListModels } from '@nemo/sdk/generated/platform/models';
import {
  ModelEntity,
  ModelEntityFilter,
  ModelEntitySortField,
  ModelEntitysPage,
} from '@nemo/sdk/generated/platform/schema';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

import { DEFAULT_NAMESPACE } from '../../constants';
import { DEFAULT_PAGE_SIZE, QUERY_PREFIX_ENTITY_STORE } from '../../constants/api';
import { isBaseModel } from '../../utils/models';

const SORT_COMPARATORS: Record<ModelEntitySortField, (a: ModelEntity, b: ModelEntity) => number> = {
  [ModelEntitySortField.name]: (a, b) => (a.name ?? '').localeCompare(b.name ?? ''),
  [ModelEntitySortField['-name']]: (a, b) => (b.name ?? '').localeCompare(a.name ?? ''),
  [ModelEntitySortField.created_at]: (a, b) =>
    (a.created_at ?? '').localeCompare(b.created_at ?? ''),
  [ModelEntitySortField['-created_at']]: (a, b) =>
    (b.created_at ?? '').localeCompare(a.created_at ?? ''),
  [ModelEntitySortField.updated_at]: (a, b) =>
    (a.updated_at ?? '').localeCompare(b.updated_at ?? ''),
  [ModelEntitySortField['-updated_at']]: (a, b) =>
    (b.updated_at ?? '').localeCompare(a.updated_at ?? ''),
};

/**
 * Widened input type for `useBaseModels({ filter })`. The generated SDK models
 * filter fields as bare scalars, but the NeMo Helix API accepts operator
 * objects on every field via the same unified filter syntax (`$like`, `$gte`,
 * etc). Coercion back to `ModelEntityFilter` happens once at the SDK boundary
 * below.
 */
export type ModelEntityFilterInput = WithFilterOperators<ModelEntityFilter>;

export interface UseBaseModelsOptions {
  workspace: string;
  filter?: ModelEntityFilterInput;
  sort?: ModelEntitySortField;
  enabled?: boolean;
}

export interface UseBaseModelsResult {
  models: ModelEntity[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  isFetchNextPageError: boolean;
  fetchNextPage: () => Promise<void>;
  refetch: () => void;
}

const getBaseModelsQueryKey = (
  workspace: string,
  filter?: ModelEntityFilterInput,
  sort?: ModelEntitySortField
) => [QUERY_PREFIX_ENTITY_STORE, 'base-models', 'infinite', workspace, filter, sort];

const getNextPageParam = (lastPage: ModelEntitysPage) => {
  if (!lastPage.pagination) return undefined;
  const { page, total_pages } = lastPage.pagination;
  return page < total_pages ? page + 1 : undefined;
};

/**
 * Fetches base models from the current workspace and the default workspace,
 * merges the results, deduplicates by name (workspace models take precedence),
 * and filters to only true base models.
 *
 * Pagination is consumer-driven via `fetchNextPage` / `hasNextPage` /
 * `isFetchingNextPage`, following the same pattern as {@link useFilesetsInfinite}.
 *
 * Queries both workspaces because base models typically live in the `default`
 * workspace (auto-discovered from providers like nvidia-build), but users may
 * also register providers in their own workspace.
 */
export function useBaseModels({
  workspace,
  filter,
  sort = ModelEntitySortField.name,
  enabled = true,
}: UseBaseModelsOptions): UseBaseModelsResult {
  const isDefaultWorkspace = workspace === DEFAULT_NAMESPACE;

  const baseQuery = {
    // Cast reflects the SDK's scalar-only modeling of filter fields; see ModelEntityFilterInput.
    filter: { base_model: false, ...filter } as ModelEntityFilter,
    page_size: DEFAULT_PAGE_SIZE,
  };

  const workspaceQuery = useInfiniteQuery({
    queryKey: getBaseModelsQueryKey(workspace, filter, sort),
    queryFn: ({ pageParam }) =>
      modelsListModels(workspace, {
        ...baseQuery,
        sort,
        page: pageParam,
      }),
    getNextPageParam,
    enabled,
    initialPageParam: 1,
  });

  const defaultQuery = useInfiniteQuery({
    queryKey: getBaseModelsQueryKey(DEFAULT_NAMESPACE, filter, sort),
    queryFn: ({ pageParam }) =>
      modelsListModels(DEFAULT_NAMESPACE, {
        ...baseQuery,
        sort,
        page: pageParam,
      }),
    getNextPageParam,
    enabled: enabled && !isDefaultWorkspace,
    initialPageParam: 1,
  });

  const hasNextPage =
    workspaceQuery.hasNextPage || (!isDefaultWorkspace && defaultQuery.hasNextPage);

  const isFetchingNextPage =
    workspaceQuery.isFetchingNextPage || (!isDefaultWorkspace && defaultQuery.isFetchingNextPage);

  const fetchNextPage = useCallback(async () => {
    const promises: Promise<unknown>[] = [];
    if (workspaceQuery.hasNextPage && !workspaceQuery.isFetchingNextPage) {
      promises.push(workspaceQuery.fetchNextPage());
    }
    if (!isDefaultWorkspace && defaultQuery.hasNextPage && !defaultQuery.isFetchingNextPage) {
      promises.push(defaultQuery.fetchNextPage());
    }
    await Promise.all(promises);
  }, [workspaceQuery, defaultQuery, isDefaultWorkspace]);

  const models = useMemo(() => {
    const workspaceModels = workspaceQuery.data?.pages.flatMap((page) => page.data) ?? [];
    const defaultModels = isDefaultWorkspace
      ? []
      : (defaultQuery.data?.pages.flatMap((page) => page.data) ?? []);

    // Workspace models take precedence
    const seen = new Set(workspaceModels.map((m) => m.name));
    const merged = [...workspaceModels, ...defaultModels.filter((m) => !seen.has(m.name))];

    // Client-side sort is required even though the backend now sorts each query:
    // we merge two independently-sorted streams (workspace + default), so the
    // concatenated result needs to be re-sorted to produce a single ordered list.
    return merged.filter(isBaseModel).sort(SORT_COMPARATORS[sort]);
  }, [workspaceQuery.data, defaultQuery.data, isDefaultWorkspace, sort]);

  return {
    models,
    isLoading: workspaceQuery.isLoading || (!isDefaultWorkspace && defaultQuery.isLoading),
    isError: workspaceQuery.isError || defaultQuery.isError,
    error: workspaceQuery.error ?? defaultQuery.error ?? null,
    hasNextPage,
    isFetchingNextPage,
    isFetchNextPageError:
      workspaceQuery.isFetchNextPageError ||
      (!isDefaultWorkspace && defaultQuery.isFetchNextPageError),
    fetchNextPage,
    refetch: () => {
      workspaceQuery.refetch();
      if (!isDefaultWorkspace) defaultQuery.refetch();
    },
  };
}
