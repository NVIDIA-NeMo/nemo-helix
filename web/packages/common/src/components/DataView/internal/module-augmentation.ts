// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DataViewColumnFilterDef } from '@nemo/common/src/components/DataView/internal/types';
import type { filterFunctions } from '@nemo/common/src/components/DataView/internal/utils/filterFunctions';
import type { Cell, RowData } from '@tanstack/react-table';

export type DataViewFilterFns = typeof filterFunctions;

/**
 * Extends the ColumnMeta interface from react-table to include the filter definition and other
 * data-view-specific column meta fields.
 * https://tanstack.com/table/latest/docs/api/core/column-def#meta
 */
declare module '@tanstack/react-table' {
  // biome-ignore lint/correctness/noUnusedVariables: type parameters are used by react-table internals
  interface ColumnMeta<TData extends RowData, TValue> {
    /** For internal use. Indicates if the column is a prebuilt column. */
    _isPrebuiltColumn?: boolean;
    /**
     * For internal use. Indicates if the initial column definition provides a size. We need this
     * because Tanstack Table will set a default size of 150px if no size is provided. If a size is
     * not provided we want to auto size the column - not use the default 150px size.
     */
    _isSizeInitialized?: boolean;
    /**
     * Controls the column cell alignment for both the header and the cell.
     *
     * To control the header separately, use `headerAlignment`.
     */
    alignment?: 'left' | 'center' | 'right';
    /**
     * Controls the header cell alignment.
     * @defaultValue "left"
     */
    headerAlignment?: 'left' | 'center' | 'right';
    /** The filter definition for the column. If provided, column filtering will be enabled. */
    filter?: DataViewColumnFilterDef<TData>;
    /**
     * By default table cells will render an OS tooltip of its children. Use this to disable the
     * tooltips, or customize the tooltip that gets generated.
     * @deprecated Replace with `title`.
     */
    tooltip?: false | ((cell: Cell<TData, TValue>) => string | undefined);
    /**
     * By default table cells will render an OS tooltip of its children. Use this to disable the
     * tooltips, or customize the tooltip that gets generated.
     */
    title?: false | ((cell: Cell<TData, TValue>) => string | undefined);
  }
}
