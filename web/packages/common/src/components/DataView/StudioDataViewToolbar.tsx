// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilterPanelToggle } from '@nemo/common/src/components/DataView/FilterPanelToggle';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal';
import { StudioAppliedFilters } from '@nemo/common/src/components/DataView/StudioAppliedFilters';
import type { ComponentProps, ReactNode } from 'react';

export interface StudioDataViewToolbarProps<DataType = unknown> {
  searchField?: string;
  showFilters: boolean;
  onToggleFilters: () => void;
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: DataView.TanstackTable.Table<DataType>;
  }) => ReactNode;
  searchBarProps?: ComponentProps<typeof DataView.SearchBar>;
  /** Rendered before the search bar, at the toolbar's leading edge. */
  slotStart?: ReactNode;
  /**
   * Additional content rendered inside the toolbar row, after the filter toggle button.
   * Use this to inject view-specific controls such as a sort dropdown.
   */
  slotEnd?: ReactNode;
}

/**
 * Toolbar for StudioDataView-style views. Renders the search bar, filter toggle button,
 * and applied filter tags. Must be used inside a `DataView.Root` context.
 *
 * Exported so it can be used in non-table views (e.g. card grids) that wrap their content
 * in a headless `DataView.Root` for filter state management.
 */
export function StudioDataViewToolbar<DataType = unknown>({
  searchField,
  showFilters,
  onToggleFilters,
  renderBulkActions,
  searchBarProps,
  slotStart,
  slotEnd,
}: StudioDataViewToolbarProps<DataType>) {
  const { table } = useInnerDataViewContext();
  const hasFilterableColumns = table.getAllLeafColumns().some((col) => col.getCanFilter());
  const hasSelectedRows = table.getSelectedRowModel().flatRows.length > 0;
  const hostsBulkActions = Boolean(renderBulkActions) && hasSelectedRows;

  if (!searchField && !hasFilterableColumns && !hostsBulkActions && !slotStart && !slotEnd)
    return null;

  return (
    <>
      <DataView.Toolbar
        slotBulkActions={
          renderBulkActions ? (
            <DataView.BulkActions>
              {({ selectedRows, table }) =>
                renderBulkActions({
                  selectedRows: selectedRows.map((row) => row.original) as DataType[],
                  table: table as DataView.TanstackTable.Table<DataType>,
                })
              }
            </DataView.BulkActions>
          ) : undefined
        }
      >
        {slotStart}
        {searchField && (
          <DataView.SearchBar
            placeholder={searchBarProps?.placeholder ?? 'Search...'}
            {...searchBarProps}
          />
        )}
        <FilterPanelToggle showFilters={showFilters} onToggle={onToggleFilters} />
        {slotEnd}
      </DataView.Toolbar>
      <StudioAppliedFilters />
    </>
  );
}
