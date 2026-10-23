// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { Close } from '@nemo/icons/icons';
import { Button, Divider, Flex, Text } from '@nvidia/foundations-react-core';
import type { Row, Table } from '@tanstack/react-table';
import { Fragment, type JSX, type ReactNode } from 'react';

export interface DataViewBulkActionsProps<TData> {
  /**
   * Function that returns a React node to render. Called with the selected rows and the table.
   *
   * @example
   * ```tsx
   * <DataView.BulkActions>
   *   {({ selectedRows, table }) => (...render your buttons here...)}
   * </DataView.BulkActions>
   * ```
   */
  children: (props: { selectedRows: Row<TData>[]; table: Table<TData> }) => ReactNode;
  onCancel?: () => void;
}

/**
 * Renders bulk actions. Must be rendered inside `DataView.Toolbar` so it can position itself
 * over the toolbar correctly.
 */
export function BulkActions<TData>({
  children,
  onCancel,
}: DataViewBulkActionsProps<TData>): JSX.Element {
  const { table } = useInnerDataViewContext();
  const selectedRows = (table as unknown as Table<TData>).getSelectedRowModel().flatRows;
  return (
    <Fragment>
      <Text kind="label/regular/md">
        {selectedRows.length} {selectedRows.length === 1 ? 'row' : 'rows'} selected
      </Text>
      <Flex align="center" gap="1">
        {children({ selectedRows, table: table as unknown as Table<TData> })}
        <Divider orientation="vertical" />
        <Button
          kind="tertiary"
          onClick={() => {
            table.resetRowSelection();
            onCancel?.();
          }}
        >
          <span className="only-mobile">
            <Close variant="line" />
          </span>
          <span className="hide-mobile">Cancel</span>
        </Button>
      </Flex>
    </Fragment>
  );
}
