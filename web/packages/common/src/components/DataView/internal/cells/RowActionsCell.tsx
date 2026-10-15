// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { Dropdown, type DropdownProps } from '@nvidia/foundations-react-core';
import type { CellContext } from '@tanstack/react-table';
import { EllipsisVertical } from 'lucide-react';
import type { JSX } from 'react';

export interface RowActionsCellProps<TData> extends Omit<DropdownProps, 'items'> {
  /** The cell context from tanstack/react-table. */
  ctx: CellContext<TData, unknown>;
  /**
   * A function that returns the items to render in the dropdown. For conditional row actions,
   * return an empty array or a falsey value to render nothing.
   */
  rowActions?: (
    row: TData,
    ctx: CellContext<TData, unknown>
  ) => DropdownProps['items'] | false | null | undefined;
}

/**
 * A cell component for the row actions column. Renders the row actions dropdown.
 * To customize the trigger, pass a custom `children` prop.
 */
export function RowActionsCell<TData>({
  children,
  ctx,
  rowActions,
  ...props
}: RowActionsCellProps<TData>): JSX.Element | null {
  const { isDataViewLoadingState, isDataViewEmptyState, isDataViewErrorState } =
    useInnerDataViewContext();
  const items = rowActions?.(ctx.row.original, ctx);
  return items && items.length > 0 ? (
    <Dropdown
      aria-label="Row Actions"
      className="-mx-[var(--table-cell-inline-padding)] -my-[var(--table-cell-block-padding)]"
      disabled={isDataViewLoadingState || isDataViewEmptyState || isDataViewErrorState}
      items={items}
      showChevron={false}
      side="left"
      {...props}
    >
      {children ?? <EllipsisVertical variant="fill" />}
    </Dropdown>
  ) : null;
}
