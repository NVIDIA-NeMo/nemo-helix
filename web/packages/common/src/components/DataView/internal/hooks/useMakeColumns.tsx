// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNumberRangeFilter } from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter/util';
import { LoadingCell } from '@nemo/common/src/components/DataView/internal/cells/LoadingCell';
import {
  RowActionsCell,
  type RowActionsCellProps,
} from '@nemo/common/src/components/DataView/internal/cells/RowActionsCell';
import {
  RowExpansionCell,
  RowExpansionHeaderCell,
} from '@nemo/common/src/components/DataView/internal/cells/RowExpansionCell';
import {
  RowSelectionCell,
  RowSelectionHeaderCell,
} from '@nemo/common/src/components/DataView/internal/cells/RowSelectionCell';
import type {
  FilterItem,
  IntentionalAny,
} from '@nemo/common/src/components/DataView/internal/types';
import type { ButtonProps, CheckboxProps, DropdownProps } from '@nvidia/foundations-react-core';
import {
  createColumnHelper,
  type CellContext,
  type Column,
  type ColumnDef,
  type ColumnHelper,
  type DisplayColumnDef,
  type FilterFn,
  type HeaderContext,
} from '@tanstack/react-table';
import { useMemo } from 'react';

const ROW_SELECTION_COLUMN_ID = 'row-selection';
const ROW_EXPANSION_COLUMN_ID = 'row-expansion';
const ROW_ACTIONS_COLUMN_ID = 'row-actions';

export const PREBUILT_COLUMN_IDS: string[] = [
  ROW_ACTIONS_COLUMN_ID,
  ROW_EXPANSION_COLUMN_ID,
  ROW_SELECTION_COLUMN_ID,
];

const DEFAULT_OPTION_BUILDER = (column: Column<IntentionalAny>): FilterItem[] =>
  Array.from(column.getFacetedUniqueValues().keys())
    .sort()
    .map((value) => ({ value: String(value) }))
    .slice(0, 500);

export function rowActionsColumn<TData>(
  options?: Partial<DisplayColumnDef<TData>> & {
    cellProps?: Partial<Omit<DropdownProps, 'items'>>;
    rowActions?: RowActionsCellProps<TData>['rowActions'];
  }
): DisplayColumnDef<TData> {
  const { rowActions, cellProps, ...overrides } = options || {};
  return {
    id: ROW_ACTIONS_COLUMN_ID,
    cell: (ctx: CellContext<TData, unknown>) => (
      <RowActionsCell ctx={ctx} rowActions={rowActions} {...cellProps} />
    ),
    header: () => <span className="sr-only">Row Actions</span>,
    enableColumnFilter: false,
    enableResizing: false,
    enableHiding: false,
    enablePinning: true,
    size: 48,
    maxSize: 48,
    minSize: 48,
    meta: {
      alignment: 'center',
      _isPrebuiltColumn: true,
      _isSizeInitialized: true,
    },
    ...overrides,
  };
}

export function rowExpansionColumn<TData>(
  options?: Partial<DisplayColumnDef<TData>> & {
    headerProps?: Partial<ButtonProps>;
    props?: Partial<ButtonProps>;
  }
): DisplayColumnDef<TData> {
  const { headerProps, props, ...overrides } = options || {};
  return {
    id: ROW_EXPANSION_COLUMN_ID,
    header: (ctx: HeaderContext<TData, unknown>) => (
      <RowExpansionHeaderCell ctx={ctx} {...headerProps} />
    ),
    cell: (ctx: CellContext<TData, unknown>) => <RowExpansionCell ctx={ctx} {...props} />,
    enableColumnFilter: false,
    enableResizing: false,
    enableHiding: false,
    enablePinning: true,
    size: 48,
    maxSize: 48,
    minSize: 48,
    meta: {
      alignment: 'center',
      _isPrebuiltColumn: true,
      _isSizeInitialized: true,
    },
    ...overrides,
  };
}

export function rowSelectionColumn<TData>(
  options?: Partial<DisplayColumnDef<TData>> & {
    headerProps?: Partial<CheckboxProps>;
    props?: Partial<CheckboxProps>;
  }
): DisplayColumnDef<TData> {
  const { headerProps, props, ...overrides } = options || {};
  return {
    id: ROW_SELECTION_COLUMN_ID,
    header: (ctx: HeaderContext<TData, unknown>) => (
      <RowSelectionHeaderCell ctx={ctx} {...headerProps} />
    ),
    cell: (ctx: CellContext<TData, unknown>) => <RowSelectionCell ctx={ctx} {...props} />,
    enableColumnFilter: false,
    enableResizing: false,
    enableHiding: false,
    enablePinning: true,
    size: 40,
    maxSize: 40,
    minSize: 40,
    meta: {
      alignment: 'center',
      _isPrebuiltColumn: true,
      _isSizeInitialized: true,
    },
    ...overrides,
  };
}

/**
 * A set of helper pre-built columns that can be used to quickly create a table.
 * Includes row actions, row expansion, and row selection columns.
 */
export const PREBUILT_COLUMNS = {
  rowActionsColumn,
  rowExpansionColumn,
  rowSelectionColumn,
};

export type PrebuiltColumns = typeof PREBUILT_COLUMNS;
export type PrebuiltColumnIds = typeof PREBUILT_COLUMN_IDS;

export type MakeColumns<TData> = (
  columnHelper: ColumnHelper<TData>,
  prebuiltColumns: typeof PREBUILT_COLUMNS
) => ColumnDef<TData, IntentionalAny>[];

export function useMakeColumns<TData>({
  makeColumns,
  overrideToLoadingCells,
}: {
  makeColumns: MakeColumns<TData>;
  overrideToLoadingCells: boolean;
}): ColumnDef<TData, IntentionalAny>[] {
  const columnHelper = useMemo(() => createColumnHelper<TData>(), []);
  return useMemo(() => {
    const builtColumns = makeColumns(columnHelper, PREBUILT_COLUMNS);
    return builtColumns.map((col) => {
      if (overrideToLoadingCells && !PREBUILT_COLUMN_IDS.includes(col.id ?? '')) {
        col.cell = LoadingCell;
      }
      if (!overrideToLoadingCells && col.meta?.filter) {
        col.enableColumnFilter = true;
        if (col.meta.filter.type === 'text' && !col.filterFn) {
          col.filterFn = 'fuzzy' as unknown as FilterFn<TData>;
        }
        // Number-range columns store `{ $gte, $lte }`. Without an explicit filterFn, TanStack
        // resolves a numeric column to `inNumberRange`, whose `autoRemove` treats that object as
        // an empty `[min, max]` tuple and silently drops the filter on every commit.
        if (isNumberRangeFilter(col.meta.filter) && !col.filterFn) {
          col.filterFn = 'numberRange' as unknown as FilterFn<TData>;
        }
        if (col.meta.filter.type === 'multi-select' || col.meta.filter.type === 'single-select') {
          if (!col.filterFn) {
            col.filterFn = (col.meta.filter.type === 'multi-select'
              ? 'multiSelect'
              : 'includesString') as unknown as FilterFn<TData>;
          }
          if (!col.meta.filter.options && !col.meta.filter.optionsBuilder) {
            col.meta.filter.optionsBuilder = DEFAULT_OPTION_BUILDER;
          }
        }
      }
      if (col.size !== undefined) {
        if (col.meta) {
          col.meta._isSizeInitialized = true;
        } else {
          col.meta = { _isSizeInitialized: true };
        }
      }
      return col;
    });
  }, [makeColumns, columnHelper, overrideToLoadingCells]);
}
