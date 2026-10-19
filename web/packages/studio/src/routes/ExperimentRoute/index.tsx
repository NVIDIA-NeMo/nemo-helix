// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import type { WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import {
  dateTimeFilter,
  type DatetimeFilterValue,
} from '@nemo/common/src/components/DataView/dateTimeFilter';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { DEFAULT_PAGE_SIZE_OPTIONS } from '@nemo/common/src/constants/pagination';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentFilter, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import {
  Block,
  Button,
  PageHeader,
  PaginationArrowButton,
  PaginationControlsGroup,
  PaginationDivider,
  PaginationItemRangeText,
  PaginationNavigationGroup,
  PaginationPageCountText,
  PaginationPageInput,
  PaginationPageSizeSelect,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { ExperimentCreateModal } from '@studio/components/ExperimentCreateModal';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { Loading } from '@studio/components/Layouts/Loading';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { ExperimentCard } from '@studio/routes/ExperimentRoute/ExperimentCard';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, type FC, useMemo, useState } from 'react';

const DEFAULT_PAGE_SIZE = 10;

/**
 * Favorites are pinned above the paginated list so they stay visible on every page. The list
 * endpoint cannot sort by `is_favorite`, so they are fetched as their own filtered query and
 * excluded from the main list to avoid rendering twice.
 *
 * The cap bounds that unpaginated query. If a workspace somehow exceeds it we stop excluding
 * favorites from the main list, so the ones past the cap stay reachable there rather than
 * disappearing from the page entirely.
 */
const MAX_PINNED_FAVORITES = 50;

/**
 * Column filters available for experiments. `created_at` / `updated_at` are base entity
 * fields the list endpoint filters via `$gte` / `$lte` ranges. They are not declared on the
 * generated `ExperimentFilter` type, so the API filter is widened here and coerced back at
 * the SDK boundary (see `filter` below).
 */
interface ExperimentColumnFilters {
  created_at?: DatetimeFilterValue;
  updated_at?: DatetimeFilterValue;
}

type ExperimentFilterInput = WithFilterOperators<ExperimentFilter> & ExperimentColumnFilters;

/**
 * Filter-only columns. They are never rendered as a table (the cards come from CustomContent);
 * they exist solely to feed `meta.filter` into the DataView filter panel and applied-filter tags.
 */
const makeFilterColumns: ComponentProps<
  typeof DataView.Root<ExperimentResponse>
>['makeColumns'] = ({ accessor }) => [
  accessor('created_at', {
    id: 'created_at',
    header: 'Created',
    enableSorting: false,
    meta: { filter: dateTimeFilter('Created At') },
  }),
  accessor('updated_at', {
    id: 'updated_at',
    header: 'Updated',
    enableSorting: false,
    meta: { filter: dateTimeFilter('Updated At') },
  }),
];

export const ExperimentRoute: FC = () => {
  useBreadcrumbs({ items: [{ slotLabel: 'Experiments' }] });

  const workspace = useWorkspaceFromPath();
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const dataViewState = useStudioDataViewState<ExperimentColumnFilters>({
    defaultPageSize: DEFAULT_PAGE_SIZE,
  });
  const page = dataViewState.pagination.state.pageIndex + 1;
  const pageSize = dataViewState.pagination.state.pageSize;
  const searchText = dataViewState.apiFilter.searchText;
  const columnFilters = dataViewState.apiFilter.filter;

  const filter = useMemo<ExperimentFilter | undefined>(() => {
    if (!searchText && !columnFilters) return undefined;
    const input: ExperimentFilterInput = {
      ...columnFilters,
      ...(searchText ? { name: { $like: searchText } } : {}),
    };
    return input as ExperimentFilter;
  }, [columnFilters, searchText]);

  // Pinned favorites: same search/column filters as the main list, narrowed to favorites, so
  // searching narrows both sections consistently.
  const { data: favoritesData } = useListExperiments(
    workspace,
    {
      page: 1,
      page_size: MAX_PINNED_FAVORITES,
      filter: { ...filter, is_favorite: true } as ExperimentFilter,
    },
    { query: { placeholderData: keepPreviousData } }
  );

  const favorites = favoritesData?.data ?? [];
  const favoritesTotal = favoritesData?.pagination?.total_results ?? 0;
  const favoritesTruncated = favoritesTotal > favorites.length;

  // `is_favorite: false` matches rows stored as false *and* legacy rows with no stored value,
  // so filtering here does not hide older experiments.
  const listFilter = useMemo<ExperimentFilter | undefined>(
    () => (favoritesTruncated ? filter : ({ ...filter, is_favorite: false } as ExperimentFilter)),
    [filter, favoritesTruncated]
  );

  const { data, isLoading, error } = useListExperiments(
    workspace,
    { page, page_size: pageSize, filter: listFilter },
    { query: { placeholderData: keepPreviousData } }
  );

  const experiments = data?.data ?? [];
  const totalResults = data?.pagination?.total_results ?? 0;

  return (
    <AccessibleTitle title="Experiments">
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0 shrink-0"
          slotHeading={
            <>
              Experiments
              <FeatureFlagBadge flag="experiment" />
            </>
          }
          slotDescription="Manage experiments for online optimization. Review runs down to the frame level."
          slotActions={
            <Button color="brand" onClick={() => setIsCreateModalOpen(true)}>
              New experiment
            </Button>
          }
        />
        <ExperimentCreateModal
          open={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          workspace={workspace}
        />
        <StudioDataView<ExperimentResponse>
          dataViewState={dataViewState}
          makeColumns={makeFilterColumns}
          searchField="name"
          attributes={{
            DataViewRoot: {
              data: experiments,
              totalCount: totalResults,
              requestStatus: error ? 'error' : isLoading ? 'loading' : undefined,
            },
            DataViewSearchBar: { placeholder: 'Search experiments...' },
          }}
        >
          <Stack className="h-full min-h-0" gap="density-md">
            <Block className="flex-1 min-h-0 overflow-auto">
              {favorites.length > 0 && (
                <Stack gap="density-md" className="pb-density-md">
                  {favorites.map((favorite) => (
                    <ExperimentCard key={favorite.id} group={favorite} workspace={workspace} />
                  ))}
                </Stack>
              )}
              <DataView.CustomContent<ExperimentResponse>
                renderLoadingState={() => <Loading description="Loading experiments..." />}
                renderEmptyState={({ hasSearchApplied, hasFiltersApplied }) => {
                  // Favorites render above and are excluded from this list, so an empty list
                  // here does not mean the workspace is empty. Suppress the first-use CTA in
                  // that case; it would contradict the experiments shown right above it.
                  if (favorites.length > 0) {
                    return (
                      <Text kind="body/regular/sm" className="text-secondary p-density-2xl">
                        {hasSearchApplied || hasFiltersApplied
                          ? 'No other experiments match the current filters.'
                          : 'No other experiments yet.'}
                      </Text>
                    );
                  }
                  return hasSearchApplied || hasFiltersApplied ? (
                    <EntityEmptyState
                      entity="experiments"
                      variant="no-results"
                      onClearFilters={dataViewState.resetFilters}
                    />
                  ) : (
                    <EntityEmptyState
                      entity="experiments"
                      variant="first-use"
                      onCreate={() => setIsCreateModalOpen(true)}
                    />
                  );
                }}
                renderErrorState={() => (
                  <ErrorPanel
                    errorMessage={getErrorMessage(
                      error ?? new Error('Failed to load experiments.')
                    )}
                  />
                )}
              >
                {({ rows }) => (
                  <Stack gap="density-md">
                    {rows.map((row) => (
                      <ExperimentCard
                        key={row.original.id}
                        group={row.original}
                        workspace={workspace}
                      />
                    ))}
                  </Stack>
                )}
              </DataView.CustomContent>
            </Block>
            <DataView.Pagination
              className="px-density-2xl py-density-lg"
              showItemsPerPage
              showWhileLessThanPageSize
              pageSizeOptions={DEFAULT_PAGE_SIZE_OPTIONS}
            >
              <>
                <PaginationControlsGroup>
                  <Text className="@max-2xl:hidden">Items per page</Text>
                  <PaginationPageSizeSelect />
                  <PaginationDivider className="@max-lg:hidden" />
                  <PaginationItemRangeText className="@max-lg:hidden" />
                </PaginationControlsGroup>
                <PaginationNavigationGroup className="gap-2">
                  <PaginationArrowButton direction="first" />
                  <PaginationArrowButton direction="previous" />
                  <PaginationPageInput />
                  <PaginationPageCountText
                    pageCountTextFormatFn={(pageMeta) => `of ${pageMeta.total}`}
                  />
                  <PaginationArrowButton direction="next" />
                  <PaginationArrowButton direction="last" />
                </PaginationNavigationGroup>
              </>
            </DataView.Pagination>
          </Stack>
        </StudioDataView>
      </Stack>
    </AccessibleTitle>
  );
};
