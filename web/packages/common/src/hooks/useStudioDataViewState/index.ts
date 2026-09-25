// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import {
  decodeColumnFiltersParam,
  FILTERS_SEARCH_PARAM,
} from '@nemo/common/src/hooks/useStudioDataViewState/columnFiltersParam';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router';
import { useDebounce } from 'use-debounce';

import * as DataView from '../../components/DataView/internal';
import { DEFAULT_PAGE, DEFAULT_PAGE_SIZE } from '../../constants/pagination';

const DEFAULT_COLUMN_PINNING = {
  left: ['row-selection'],
  right: ['row-actions'],
};

/**
 * Extract the options type from DataView.useDataViewState (excluding undefined)
 */
type DataViewStateOptions = NonNullable<Parameters<typeof DataView.useDataViewState>[0]>;

/**
 * Options for useStudioDataViewState hook.
 * Accepts the same options as DataView.useDataViewState, plus URL sync defaults.
 * Note: `pagination` and `sorting` are read from URL params, not from options.
 */
export interface UseStudioDataViewStateOptions extends Omit<
  DataViewStateOptions,
  'pagination' | 'sorting'
> {
  /** Default page number (1-based) when not specified in URL. Defaults to 1. */
  defaultPage?: number;
  /** Default page size when not specified in URL. Defaults to 50. */
  defaultPageSize?: number;
  /**
   * Default sorting when not specified in URL. Always an ordered list — a single-column default is
   * just an array of one. Uses the same format as DataView's sorting state.
   * Example: [{ id: 'created_at', desc: true }] for descending by created_at.
   */
  defaultSort?: SortEntry[];
  /**
   * Enable multi-column sort. When true, the `sort` URL param and the sorting state round-trip an
   * ordered, comma-separated list of fields (e.g. shift-click a second column header). Defaults to
   * false — single-sort, exactly as before.
   */
  multiSort?: boolean;
  /** Maps a filter column id to the API key it's emitted under (id used as-is when absent). Also
   * accepts a function `(id) => key | undefined` for dynamic ids, e.g. `latency_ms`→`latency_ms.mean`. */
  filterFieldMap?: Record<string, string> | ((id: string) => string | undefined);
}

interface SortEntry {
  id: string;
  desc: boolean;
}

// Stable empty-sort reference so the no-default case doesn't churn `urlSorting`'s memo each render.
const EMPTY_SORT: SortEntry[] = [];

const sortEntryToString = (entry: SortEntry): string => (entry.desc ? `-${entry.id}` : entry.id);

/** Encode sorting state to the URL `sort` value. Single-sort keeps only the primary column; multi-sort
 * emits the full ordered, comma-separated list. Returns null for no sort. */
const encodeSorting = (sortingState: SortEntry[], multiSort: boolean): string | null => {
  const active = multiSort ? sortingState : sortingState.slice(0, 1);
  return active.length ? active.map(sortEntryToString).join(',') : null;
};

/** Parse the URL `sort` value into ordered sort entries. Single-sort reads only the first field. */
const decodeSorting = (sortParam: string | null, multiSort: boolean): SortEntry[] => {
  if (!sortParam) return [];
  // Always split on comma so a stray multi-field URL is parsed field-by-field; in single-sort mode
  // keep only the first field rather than treating the whole comma string as one (malformed) id.
  const tokens = sortParam
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean);
  const used = multiSort ? tokens : tokens.slice(0, 1);
  return used.map((t) => ({ id: t.startsWith('-') ? t.slice(1) : t, desc: t.startsWith('-') }));
};

/**
 * API filter object exposed by the hook. The `filter` shape is parameterized by
 * `FilterType` so consumers get type-checked filter keys and values without
 * needing `as` casts at the call site.
 *
 * Defaults to `Record<string, unknown>` for callers that don't supply a type.
 */
export interface ApiFilter<FilterType = Record<string, unknown>> {
  searchText?: string;
  filter?: Partial<FilterType>;
}

/**
 * Extended DataView state that includes helper functions for common table operations.
 */
export interface StudioDataViewState<FilterType = Record<string, unknown>>
  extends DataView.DataViewState {
  /**
   * Resets pagination to page 1 and clears row selection.
   * Call this when search/filter criteria change to ensure users see results from the beginning.
   */
  resetPagination: () => void;
  /** Debounced search bar value (300ms). Use this for API queries. */
  debouncedSearchBar: string;
  /** Debounced column filters (300ms). Use this for API queries. */
  debouncedColumnFilters: DataView.TanstackTable.ColumnFiltersState;
  /**
   * Convention-mapped API filter object built from debounced columnFilters and searchBar.
   * - `columnFilters` entries map to `filter` keys: `{id, value}` → `filter[id] = value`,
   *   unless a `filterFieldMap` entry remaps the id to a different API key.
   * - `searchBar` is exposed as `searchText` when non-empty. Consumers are responsible for
   *   mapping `searchText` onto the appropriate filter field (and wrapping it in an operator
   *   like `$like` if fuzzy matching is desired).
   */
  apiFilter: ApiFilter<FilterType>;
  /** Clears searchBar, columnFilters, and resets pagination. */
  resetFilters: () => void;
}

/**
 * Opinionated React hook that wraps {@link DataView.useDataViewState} to synchronize
 * table state (pagination and sorting) with URL search parameters.
 *
 * Ensures DataView's pagination and sorting state and the URL's query params are always in sync,
 * so that users can share/bookmark current view state, and can use browser navigation
 * controls (back/forward) seamlessly.
 *
 * @param {UseStudioDataViewStateOptions} [options] - Accepts all options for DataView's useDataViewState
 *   except `pagination` and `sorting`, plus `defaultPage`, `defaultPageSize`, and `defaultSort`
 *   for setting fallback URL values.
 *
 * @returns {StudioDataViewState} - Extended DataView state object with URL sync and helper functions.
 *
 * @remarks
 * - Page numbers in the URL are 1-based, while DataView expects 0-based pageIndex.
 * - Sort in URL uses string format: "field" for ascending, "-field" for descending.
 * - defaultSort uses DataView's object format: { id: 'field', desc: boolean }.
 * - Pagination automatically resets to page 1 when sorting changes.
 * - Use `resetPagination()` when search/filter criteria change.
 * - Intended for table/data grid views where browser-driven navigation is desirable.
 */
export const useStudioDataViewState = <FilterType = Record<string, unknown>>(
  options?: UseStudioDataViewStateOptions
): StudioDataViewState<FilterType> => {
  const {
    defaultPage = DEFAULT_PAGE,
    defaultPageSize = DEFAULT_PAGE_SIZE,
    defaultSort,
    multiSort = false,
    filterFieldMap,
    ...dataViewOptions
  } = options ?? {};

  const [searchParams, setSearchParams] = useSearchParams();

  // Parse pagination from URL
  const parsePageParam = (value: string | null, defaultValue: number): number => {
    if (!value) return defaultValue;
    const parsed = parseInt(value, 10);
    return isNaN(parsed) || parsed < 1 ? defaultValue : parsed;
  };

  const urlPage = parsePageParam(searchParams.get('page'), defaultPage);
  const urlPageSize = parsePageParam(searchParams.get('page_size'), defaultPageSize);

  // Parse sort from URL, falling back to defaultSort
  const sortParam = searchParams.get('sort');
  const defaultSortEntries = defaultSort ?? EMPTY_SORT;
  const urlSorting = useMemo(() => {
    const decoded = decodeSorting(sortParam, multiSort);
    return decoded.length ? decoded : defaultSortEntries;
  }, [sortParam, defaultSortEntries, multiSort]);

  // Convert sorting object to URL string format
  const defaultSortString = defaultSortEntries.length
    ? encodeSorting(defaultSortEntries, multiSort)
    : null;

  // Update URL params helper
  const updateUrlParams = useCallback(
    ({
      page,
      pageSize,
      sort,
      search,
      filters,
    }: {
      page?: number;
      pageSize?: number;
      sort?: string | null;
      search?: string;
      filters?: string | null;
    }) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);

          // Handle page
          if (page !== undefined) {
            if (page === defaultPage) {
              next.delete('page');
            } else {
              next.set('page', String(page));
            }
          }

          // Handle page_size
          if (pageSize !== undefined) {
            if (pageSize === defaultPageSize) {
              next.delete('page_size');
            } else {
              next.set('page_size', String(pageSize));
            }
          }

          // Handle sort
          if (sort !== undefined) {
            if (sort === null || sort === defaultSortString) {
              next.delete('sort');
            } else {
              next.set('sort', sort);
            }
          }

          // Handle search
          if (search !== undefined) {
            if (search === '') {
              next.delete('s');
            } else {
              next.set('s', search);
            }
          }

          // Handle filters
          if (filters !== undefined) {
            if (filters === null) {
              next.delete(FILTERS_SEARCH_PARAM);
            } else {
              next.set(FILTERS_SEARCH_PARAM, encodeURIComponent(filters));
            }
          }

          return next;
        },
        { replace: true }
      );
    },
    [setSearchParams, defaultPage, defaultPageSize, defaultSortString]
  );

  // Parse search and filters from URL
  const urlSearch = searchParams.get('s') ?? '';
  const urlFiltersParam = searchParams.get(FILTERS_SEARCH_PARAM);
  const urlColumnFilters = useMemo<DataView.TanstackTable.ColumnFiltersState>(
    () => decodeColumnFiltersParam(urlFiltersParam),
    [urlFiltersParam]
  );

  // Initialize DataView with pagination, sorting, search, and filters from URL.
  // Default column pinning is set first so consumers can override via options.
  const dataViewState = DataView.useDataViewState({
    columnPinning: DEFAULT_COLUMN_PINNING,
    ...dataViewOptions,
    pagination: {
      pageIndex: urlPage - 1,
      pageSize: urlPageSize,
    },
    sorting: urlSorting,
    searchBar: urlSearch,
    columnFilters: urlColumnFilters,
  });

  // Track previous DataView state to detect internal changes
  const prevPaginationRef = useRef(dataViewState.pagination.state);
  const prevSortingRef = useRef(dataViewState.sorting.state);

  // Track pending URL updates to avoid reverting changes before URL has updated
  // This prevents the race condition where pagination.set() triggers a re-render
  // before searchParams has updated, causing the old URL values to sync back.
  const pendingSortRef = useRef<string | null | undefined>(undefined);
  const pendingPageRef = useRef<number | undefined>(undefined);

  /**
   * Resets pagination to page 1 and clears row selection.
   * Use this when search/filter criteria change.
   */
  const resetPagination = useCallback(() => {
    const { pageIndex, pageSize } = dataViewState.pagination.state;

    // Only reset if not already on page 0
    if (pageIndex !== 0) {
      prevPaginationRef.current = { pageIndex: 0, pageSize };
      dataViewState.pagination.set((prev: { pageIndex: number; pageSize: number }) => ({
        ...prev,
        pageIndex: 0,
      }));
    }

    // Clear row selection
    dataViewState.rowSelection.set({});
  }, [dataViewState.pagination, dataViewState.rowSelection]);

  // Bidirectional pagination sync: DataView ↔ URL
  // Priority: DataView changes sync to URL first. Only if DataView didn't change,
  // then URL changes sync to DataView (e.g., browser back/forward).
  // This prevents infinite loops by checking which side changed first.
  useEffect(() => {
    const { pageIndex, pageSize } = dataViewState.pagination.state;
    const prev = prevPaginationRef.current;

    const dataViewChanged = pageIndex !== prev.pageIndex || pageSize !== prev.pageSize;

    if (dataViewChanged) {
      // DataView changed → sync to URL
      prevPaginationRef.current = { pageIndex, pageSize };
      pendingPageRef.current = pageIndex + 1;
      updateUrlParams({ page: pageIndex + 1, pageSize });
    } else {
      // Clear pending if URL has caught up
      if (pendingPageRef.current === urlPage) {
        pendingPageRef.current = undefined;
      }

      // DataView didn't change → check if URL changed (e.g., browser back/forward)
      // Skip if we have a pending update that hasn't been reflected in URL yet
      if (pendingPageRef.current !== undefined) {
        return;
      }

      const targetPageIndex = urlPage - 1;
      const urlDiffersFromDataView = targetPageIndex !== pageIndex || urlPageSize !== pageSize;

      if (urlDiffersFromDataView) {
        // URL changed externally → sync to DataView
        prevPaginationRef.current = { pageIndex: targetPageIndex, pageSize: urlPageSize };
        dataViewState.pagination.set({ pageIndex: targetPageIndex, pageSize: urlPageSize });
      }
    }
  }, [dataViewState.pagination, urlPage, urlPageSize, updateUrlParams]);

  // Bidirectional sorting sync: DataView ↔ URL
  // Same priority logic as pagination.
  // Also resets pagination to page 1 when sorting changes (common UX pattern).
  useEffect(() => {
    const sortingState = dataViewState.sorting.state;
    // Compare via the encoded URL string: with multiSort=false this collapses to the primary column
    // exactly as before; with multiSort=true it tracks the full ordered list.
    const currentSortString = encodeSorting(sortingState, multiSort);
    const prevSortString = encodeSorting(prevSortingRef.current, multiSort);

    const dataViewSortChanged = currentSortString !== prevSortString;

    if (dataViewSortChanged) {
      // DataView sorting changed → sync to URL
      prevSortingRef.current = [...sortingState];
      pendingSortRef.current = currentSortString;

      // Reset pagination to page 1 when sorting changes (common UX pattern)
      // Update both sort AND page in a single URL update to avoid race conditions
      const { pageIndex, pageSize } = dataViewState.pagination.state;
      const needsPageReset = pageIndex !== 0;

      if (needsPageReset) {
        prevPaginationRef.current = { pageIndex: 0, pageSize };
        pendingPageRef.current = 1;
        updateUrlParams({ sort: currentSortString, page: 1 });
        dataViewState.pagination.set((prev: { pageIndex: number; pageSize: number }) => ({
          ...prev,
          pageIndex: 0,
        }));
      } else {
        updateUrlParams({ sort: currentSortString });
      }
    } else {
      // Clear pending if URL has caught up
      const urlSortString = encodeSorting(urlSorting, multiSort);
      if (pendingSortRef.current === urlSortString) {
        pendingSortRef.current = undefined;
      }

      // DataView didn't change → check if URL sorting changed
      // Skip if we have a pending update that hasn't been reflected in URL yet
      if (pendingSortRef.current !== undefined) {
        return;
      }

      if (urlSortString !== currentSortString) {
        // URL sorting changed externally → sync to DataView
        prevSortingRef.current = [...urlSorting];
        dataViewState.sorting.set([...urlSorting]);
      }
    }
  }, [dataViewState.sorting, dataViewState.pagination, urlSorting, updateUrlParams, multiSort]);

  // Debounced search bar and column filters for API queries
  const [debouncedSearchBar] = useDebounce(dataViewState.searchBar.state, DEFAULT_DEBOUNCE_MS);
  const [debouncedColumnFilters] = useDebounce(
    dataViewState.columnFiltering.state,
    DEFAULT_DEBOUNCE_MS
  );

  // Track previous debounced values to detect changes for pagination reset
  const prevDebouncedSearchRef = useRef(debouncedSearchBar);
  const prevDebouncedFiltersRef = useRef(debouncedColumnFilters);

  // Auto-reset pagination when search or filters change
  useEffect(() => {
    const searchChanged = prevDebouncedSearchRef.current !== debouncedSearchBar;
    const filtersChanged = prevDebouncedFiltersRef.current !== debouncedColumnFilters;

    prevDebouncedSearchRef.current = debouncedSearchBar;
    prevDebouncedFiltersRef.current = debouncedColumnFilters;

    if (searchChanged || filtersChanged) {
      resetPagination();
    }
  }, [debouncedSearchBar, debouncedColumnFilters, resetPagination]);

  // Bidirectional search bar sync: DataView ↔ URL
  const prevSearchBarRef = useRef(dataViewState.searchBar.state);
  const pendingSearchRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    const currentSearch = dataViewState.searchBar.state;
    const dataViewChanged = currentSearch !== prevSearchBarRef.current;

    if (dataViewChanged) {
      prevSearchBarRef.current = currentSearch;
      pendingSearchRef.current = currentSearch;
      updateUrlParams({ search: currentSearch });
    } else {
      if (pendingSearchRef.current === urlSearch) {
        pendingSearchRef.current = undefined;
      }
      if (pendingSearchRef.current !== undefined) return;

      if (urlSearch !== currentSearch) {
        prevSearchBarRef.current = urlSearch;
        dataViewState.searchBar.set(urlSearch);
      }
    }
  }, [dataViewState.searchBar, urlSearch, updateUrlParams]);

  // Bidirectional column filters sync: DataView ↔ URL
  const prevColumnFiltersRef = useRef(dataViewState.columnFiltering.state);
  const pendingFiltersRef = useRef<string | undefined>(undefined);

  useEffect(() => {
    const currentFilters = dataViewState.columnFiltering.state;
    const dataViewChanged = currentFilters !== prevColumnFiltersRef.current;

    if (dataViewChanged) {
      prevColumnFiltersRef.current = currentFilters;
      const filtersJson = currentFilters.length > 0 ? JSON.stringify(currentFilters) : null;
      pendingFiltersRef.current = filtersJson ?? undefined;
      updateUrlParams({ filters: filtersJson });
    } else {
      const currentUrlFiltersJson = urlFiltersParam
        ? decodeURIComponent(urlFiltersParam)
        : undefined;
      if (pendingFiltersRef.current === currentUrlFiltersJson) {
        pendingFiltersRef.current = undefined;
      }
      if (pendingFiltersRef.current !== undefined) return;

      const urlDiffersFromDataView =
        JSON.stringify(currentFilters) !== JSON.stringify(urlColumnFilters);

      if (urlDiffersFromDataView) {
        prevColumnFiltersRef.current = urlColumnFilters;
        dataViewState.columnFiltering.set(urlColumnFilters);
      }
    }
  }, [dataViewState.columnFiltering, urlColumnFilters, urlFiltersParam, updateUrlParams]);

  // Convention-mapped API filter object.
  // The narrowing from TanStack's untyped `ColumnFiltersState` ({id: string, value: unknown}[])
  // to `Partial<FilterType>` happens here in one place, so call sites get a typed
  // `apiFilter.filter` and don't need their own `as` casts.
  const apiFilter = useMemo<ApiFilter<FilterType>>(() => {
    const result: ApiFilter<FilterType> = {};

    if (debouncedSearchBar) {
      result.searchText = debouncedSearchBar;
    }

    if (debouncedColumnFilters.length > 0) {
      result.filter = Object.fromEntries(
        debouncedColumnFilters
          .filter((f) => {
            if (f.value === undefined || f.value === '') return false;
            if (
              typeof f.value === 'object' &&
              f.value !== null &&
              Object.keys(f.value).length === 0
            )
              return false;
            return true;
          })
          .map((f) => {
            const mappedKey =
              typeof filterFieldMap === 'function' ? filterFieldMap(f.id) : filterFieldMap?.[f.id];
            return [mappedKey ?? f.id, f.value];
          })
      ) as Partial<FilterType>;
    }

    return result;
  }, [debouncedSearchBar, debouncedColumnFilters, filterFieldMap]);

  // Reset all filters and search
  const resetFilters = useCallback(() => {
    dataViewState.searchBar.set('');
    dataViewState.columnFiltering.set([]);
    resetPagination();
  }, [dataViewState.searchBar, dataViewState.columnFiltering, resetPagination]);

  return {
    ...dataViewState,
    resetPagination,
    debouncedSearchBar,
    debouncedColumnFilters,
    apiFilter,
    resetFilters,
  };
};
