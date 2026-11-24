// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The variant half of `EntityEmptyStateProps`, as a data view's empty state picks it. */
export type DataViewEmptyStateVariant =
  | { variant: 'first-use' }
  | { variant: 'no-results'; onClearFilters: () => void };

/**
 * Picks the `EntityEmptyState` variant for a `StudioDataView`'s `renderEmptyState`:
 * `no-results` (with a way to clear) when a filter or search is hiding rows,
 * `first-use` when there is genuinely nothing yet. Spread the result into
 * `EntityEmptyState` alongside `entity` and `workspace`.
 */
export const getDataViewEmptyStateVariant = (
  {
    hasFiltersApplied,
    hasSearchApplied,
  }: { hasFiltersApplied: boolean; hasSearchApplied: boolean },
  onClearFilters: () => void
): DataViewEmptyStateVariant =>
  hasFiltersApplied || hasSearchApplied
    ? { variant: 'no-results', onClearFilters }
    : { variant: 'first-use' };
