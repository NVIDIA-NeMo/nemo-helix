// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Checkbox } from '@nvidia/foundations-react-core';
import { ArrowDown, ArrowUp } from '@nvidia/react-gui-icons/icons';
import type { SortOrder } from '@studio/components/filesets/FilesetFileExplorer/hooks/useFileActions';
import type { ExtraColumn } from '@studio/components/filesets/FilesetFileExplorer/types';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { useMemo } from 'react';

export interface UseFilesetFileExplorerColumnsOptions {
  selectedItems: FileSystemNode[];
  rowContents: FileSystemNode[];
  selectAllItems: () => void;
  clearSelectedItems: () => void;
  sortFiles: (sortBy: 'name' | 'size') => void;
  sortOrder: SortOrder;
  extraColumns?: ExtraColumn[];
}

export function useFilesetFileExplorerColumns({
  selectedItems,
  rowContents,
  selectAllItems,
  clearSelectedItems,
  sortFiles,
  sortOrder,
  extraColumns,
}: UseFilesetFileExplorerColumnsOptions) {
  return useMemo(
    () => [
      {
        children: (
          <Checkbox
            checked={
              selectedItems.length === rowContents.length
                ? true
                : selectedItems.length > 0
                  ? 'indeterminate'
                  : false
            }
            onCheckedChange={(checked) => {
              if (checked) {
                selectAllItems();
              } else {
                clearSelectedItems();
              }
            }}
            attributes={{
              CheckboxInput: {
                'aria-label': `Select all files and directories`,
                'aria-labelledby': undefined,
              },
            }}
          />
        ),
        attributes: {
          TableHeaderCell: { style: { width: 48 } },
        },
      },
      {
        children: (
          <Button type="button" kind="tertiary" onClick={() => sortFiles('name')}>
            Name
            {sortOrder.sortBy === 'name' &&
              (sortOrder.order === 'asc' ? <ArrowUp /> : <ArrowDown />)}
          </Button>
        ),
      },
      {
        children: (
          <Button type="button" kind="tertiary" onClick={() => sortFiles('size')}>
            Size
            {sortOrder.sortBy === 'size' &&
              (sortOrder.order === 'asc' ? <ArrowUp /> : <ArrowDown />)}
          </Button>
        ),
      },
      ...(extraColumns ?? []).map((col) => ({
        children: col.header,
        attributes:
          col.width !== undefined
            ? { TableHeaderCell: { style: { width: col.width } } }
            : undefined,
      })),
      {
        children: <></>,
        attributes: {
          TableHeaderCell: { style: { width: 58 } },
        },
      },
    ],
    [
      selectedItems,
      rowContents,
      selectAllItems,
      clearSelectedItems,
      sortFiles,
      sortOrder,
      extraColumns,
    ]
  );
}
