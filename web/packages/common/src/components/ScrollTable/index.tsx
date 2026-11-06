// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import {
  Pagination,
  Panel,
  Skeleton,
  TableBody,
  type TableColumnDefinition,
  TableDataCell,
  TableHead,
  TableHeaderCell,
  type TableProps,
  TableRoot,
  TableRow,
  type TableRowDefinition,
} from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { type ComponentProps, type FC, type ReactNode, useMemo } from 'react';

export type PaginatedTableProps = {
  loading?: boolean;
  columns: TableColumnDefinition[];
  rows: TableRowDefinition[];
  pagination?: boolean;
  className?: string;
  slotHeader?: ReactNode;
  slotFooter?: ReactNode;
  slotEmptyState?: ReactNode;
  tableProps?: Omit<TableProps, 'columns' | 'rows' | 'layout' | 'align'>;
  allowHorizontalScroll?: boolean;
  paginationProps?: ComponentProps<typeof Pagination>;
};
export const ScrollTable: FC<PaginatedTableProps> = ({
  paginationProps,
  loading,
  columns,
  rows,
  pagination,
  className,
  slotHeader,
  slotFooter,
  tableProps,
  allowHorizontalScroll,
  slotEmptyState,
}) => {
  const displayRows = useMemo((): TableRowDefinition[] => {
    if (loading) {
      return Array.from({ length: paginationProps?.pageSize || 10 }, (_, index) => ({
        id: `skeleton-${index}`,
        cells: Array.from({ length: columns.length || 5 }, () => ({
          children: <Skeleton animated />,
        })),
      }));
    }

    return rows || [];
  }, [loading, rows, paginationProps?.pageSize, columns]);

  const getCellContent = (cell: TableRowDefinition['cells'][number]) => {
    if (typeof cell === 'string') {
      return cell;
    }
    return cell.children;
  };

  const getCellAttributes = (cell: TableRowDefinition['cells'][number]) => {
    if (typeof cell === 'string') {
      return undefined;
    }
    return cell.attributes?.TableDataCell;
  };

  const tableClassName = cn('bg-inherit', {
    'w-max-content min-w-full': allowHorizontalScroll,
    'w-full': !allowHorizontalScroll,
  });

  return (
    <Panel
      elevation="high"
      density="compact"
      className={cn('h-full w-full overflow-hidden py-density-lg', className)}
      attributes={{
        PanelContent: {
          className: 'flex-1 min-h-0 w-full overflow-auto',
        },
      }}
      slotFooter={
        <>
          {pagination && (
            <Pagination
              className="flex-none"
              kind="tabs"
              {...(paginationProps
                ? (() => {
                    return {
                      ...paginationProps,
                      displayControls: undefined,
                    };
                  })()
                : {})}
              displayControls
              totalItems={paginationProps?.totalItems || 0}
            />
          )}
          {slotFooter}
        </>
      }
    >
      {slotHeader}
      <TableRoot
        className={cn(tableClassName, { 'h-full': !loading && displayRows.length === 0 })}
        layout="fixed"
        align="left"
        {...tableProps}
      >
        <TableHead>
          <TableRow>
            {columns.map((column, index) => (
              <TableHeaderCell key={index} {...column.attributes?.TableHeaderCell}>
                {column.children}
              </TableHeaderCell>
            ))}
          </TableRow>
        </TableHead>
        {!loading && displayRows.length === 0 ? (
          <TableBody className="h-full">
            <TableRow className="h-full">
              <TableDataCell colSpan={columns.length} className="h-full">
                {slotEmptyState ?? (
                  <TableEmptyState
                    className="py-4"
                    header="No Entries Found"
                    emptyMessage="No entries available."
                  />
                )}
              </TableDataCell>
            </TableRow>
          </TableBody>
        ) : (
          <TableBody>
            {displayRows.map((row) => (
              <TableRow
                key={row.id}
                onClick={row.onRowSelect ? () => row.onRowSelect?.({ rowId: row.id }) : undefined}
                {...row.attributes?.TableRow}
              >
                {row.cells.map((cell, cellIndex) => (
                  <TableDataCell key={cellIndex} {...getCellAttributes(cell)}>
                    {getCellContent(cell)}
                  </TableDataCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        )}
      </TableRoot>
    </Panel>
  );
};
