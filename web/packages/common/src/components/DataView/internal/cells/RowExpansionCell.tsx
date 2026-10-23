// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { ChevronDown, ChevronUp, ChevronDoubleDown, ChevronDoubleUp } from '@nemo/icons/icons';
import { Button, type ButtonProps } from '@nvidia/foundations-react-core';
import type { CellContext, HeaderContext } from '@tanstack/react-table';
import type { JSX } from 'react';

export interface RowExpansionCellProps<TData> extends ButtonProps {
  /** The cell context from tanstack/react-table. */
  ctx: CellContext<TData, unknown>;
}

/**
 * A cell component for the row expansion column. Renders the row expansion button.
 * Only renders if the row is expandable.
 */
export function RowExpansionCell<TData>({
  children,
  ctx,
  ...props
}: { ctx: CellContext<TData, unknown> } & ButtonProps): false | JSX.Element {
  const { isDataViewLoadingState, isDataViewEmptyState, isDataViewErrorState } =
    useInnerDataViewContext();
  const expanded = ctx.row.getIsExpanded();
  return (
    (ctx.row.getCanExpand() || isDataViewLoadingState) && (
      <Button
        aria-label={expanded ? 'Collapse row' : 'Expand row'}
        className="-mx-[var(--table-cell-inline-padding)] -my-[var(--table-cell-block-padding)]"
        data-expanded={expanded}
        disabled={isDataViewLoadingState || isDataViewEmptyState || isDataViewErrorState}
        kind="tertiary"
        onClick={ctx.row.getToggleExpandedHandler()}
        {...props}
      >
        {children ?? (expanded ? <ChevronUp /> : <ChevronDown />)}
      </Button>
    )
  );
}

export interface RowExpansionHeaderCellProps<TData> extends ButtonProps {
  /** The header context from tanstack/react-table. */
  ctx: HeaderContext<TData, unknown>;
}

/**
 * A header cell component for the row expansion column.
 * Only renders if the table has any expandable rows.
 */
export function RowExpansionHeaderCell<TData>({
  children,
  ctx,
  ...props
}: RowExpansionHeaderCellProps<TData>): false | JSX.Element {
  const { isDataViewLoadingState, isDataViewEmptyState, isDataViewErrorState } =
    useInnerDataViewContext();
  const isAllRowsExpanded = ctx.table.getIsAllRowsExpanded();
  return (
    ctx.table.getCanSomeRowsExpand() && (
      <Button
        aria-label={isAllRowsExpanded ? 'Collapse all rows' : 'Expand all rows'}
        className="-mx-(--table-cell-inline-padding) -my-(--table-cell-block-padding)"
        data-all-expanded={isAllRowsExpanded}
        disabled={isDataViewLoadingState || isDataViewEmptyState || isDataViewErrorState}
        kind="tertiary"
        onClick={ctx.table.getToggleAllRowsExpandedHandler()}
        {...props}
      >
        {children ?? (isAllRowsExpanded ? <ChevronDoubleUp /> : <ChevronDoubleDown />)}
      </Button>
    )
  );
}
