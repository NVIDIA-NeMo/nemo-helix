// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { Tag, Text } from '@nvidia/foundations-react-core';
import { formatCellValue } from '@studio/components/FileRowEditor/schema';
import type { DataFileColumn, DataFileRow } from '@studio/components/FileRowEditor/types';
import type { ComponentProps } from 'react';

type MakeColumns = ComponentProps<typeof StudioDataView<DataFileRow>>['makeColumns'];

export interface DataFileColumnHandlers {
  onEdit: (row: DataFileRow) => void;
  onDuplicate: (row: DataFileRow) => void;
  onDelete: (row: DataFileRow) => void;
}

/** Renders a single cell's value according to its inferred logical type. */
const renderCell = (value: unknown, column: DataFileColumn) => {
  if (value === null || value === undefined) {
    return <Text className="text-disabled">—</Text>;
  }
  switch (column.type) {
    case 'boolean':
      return (
        <Tag kind="outline" color={value ? 'green' : 'gray'} readOnly>
          {String(value)}
        </Tag>
      );
    case 'int':
    case 'float':
      return <Text className="font-mono text-[12px]">{String(value)}</Text>;
    case 'json':
      return (
        <Text className="font-mono text-[12px] text-secondary">
          {formatCellValue(value, 'json')}
        </Text>
      );
    default:
      return <Text className="text-secondary">{String(value)}</Text>;
  }
};

/**
 * Builds the StudioDataView column set for a data file from its inferred schema.
 * Columns are derived from {@link columns} rather than hard-coded, so the table works
 * for any row-like shape. Sorting is enabled for scalar columns; `json` columns get a
 * stringified preview. Low-cardinality columns expose a single-select filter.
 */
export const makeDataFileColumns =
  (
    columns: DataFileColumn[],
    { onEdit, onDuplicate, onDelete }: DataFileColumnHandlers
  ): MakeColumns =>
  ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
    rowSelectionColumn({ size: 48 }),
    ...columns.map((column) =>
      accessor((row) => row[column.key], {
        id: column.key,
        header: column.label,
        enableSorting: column.type !== 'json',
        cell: ({ getValue }) => renderCell(getValue(), column),
        meta: column.options
          ? {
              filter: {
                type: 'single-select',
                label: column.label,
                options: [
                  { value: '', label: 'All' },
                  ...column.options.map((value) => ({ value, label: value })),
                ],
              },
            }
          : undefined,
      })
    ),
    rowActionsColumn({
      size: 58,
      enableResizing: false,
      cell: ({ row: { original: row } }) => (
        <QuickActionsMenuRoot
          actions={[
            { label: 'Edit row', onSelect: () => onEdit(row) },
            { label: 'Duplicate row', onSelect: () => onDuplicate(row) },
          ]}
          deleteAction={{ label: 'Delete row', onSelect: () => onDelete(row) }}
        />
      ),
    }),
  ];
