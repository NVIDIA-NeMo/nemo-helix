// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IconDocument, FolderClosed, FolderOpen } from '@nemo/icons/icons';
import {
  Checkbox,
  Flex,
  ProgressBar,
  type TableRowDefinition,
} from '@nvidia/foundations-react-core';
import {
  INDENT_PER_LEVEL,
  PENDING_FILE_OID,
} from '@studio/components/filesets/FilesetFileExplorer/constants';
import { getItemId } from '@studio/components/filesets/FilesetFileExplorer/helpers';
import type { ExtraColumn } from '@studio/components/filesets/FilesetFileExplorer/types';
import { DirectoryQuickActions } from '@studio/components/FilesTable/DirectoryQuickActions';
import { FileQuickActions } from '@studio/components/FilesTable/FileQuickActions';
import type { FileSystemNode, TreeRow } from '@studio/components/FilesTable/utils';
import { getFolderSize, getHumanReadableFileSize } from '@studio/util/files';
import { useMemo } from 'react';

export interface UseFilesetFileExplorerRowsOptions {
  treeRows: TreeRow[];
  expandedFolders: Set<string>;
  handleUserFolderToggle: (path: string) => void;
  toggleFolderSubtreeExpand: (path: string) => void;
  isFolderSubtreeExpanded: (path: string) => boolean;
  folderHasSubfolders: (path: string) => boolean;
  datasetId: string;
  currentFolder?: string;
  onFileSelect?: (filePath: string) => void;
  isReadWriteDataset: boolean;
  selectedItems: FileSystemNode[];
  addSelectedItem: (item: FileSystemNode) => void;
  removeSelectedItem: (item: FileSystemNode) => void;
  searchQuery: string;
  extraColumns?: ExtraColumn[];
}

export function useFilesetFileExplorerRows({
  treeRows,
  expandedFolders,
  handleUserFolderToggle,
  toggleFolderSubtreeExpand,
  isFolderSubtreeExpanded,
  folderHasSubfolders,
  datasetId,
  currentFolder,
  onFileSelect,
  isReadWriteDataset,
  selectedItems,
  addSelectedItem,
  removeSelectedItem,
  searchQuery,
  extraColumns,
}: UseFilesetFileExplorerRowsOptions): TableRowDefinition[] {
  return useMemo(
    () =>
      treeRows.map(({ node, depth }) => ({
        id: getItemId(node),
        cells: [
          {
            children: (
              <Checkbox
                disabled={node.oid === PENDING_FILE_OID}
                checked={selectedItems.includes(node)}
                onCheckedChange={(checked) => {
                  if (checked) {
                    addSelectedItem(node);
                  } else {
                    removeSelectedItem(node);
                  }
                }}
                attributes={{
                  CheckboxInput: {
                    'aria-label': `Select path ${node.path}`,
                    'aria-labelledby': undefined,
                  },
                }}
              />
            ),
          },
          {
            children: (
              <Flex direction="col" gap="density-xs">
                {/* eslint-disable-next-line no-restricted-syntax -- dynamic tree indent */}
                <div style={{ paddingLeft: depth * INDENT_PER_LEVEL }}>
                  <Flex gap="density-sm" align="center">
                    {node.type === 'directory' ? (
                      expandedFolders.has(node.path) ? (
                        <FolderOpen className="shrink-0" />
                      ) : (
                        <FolderClosed className="shrink-0" />
                      )
                    ) : (
                      <IconDocument className="shrink-0" />
                    )}
                    <div>{searchQuery ? node.path : node.path.split('/').pop()}</div>
                  </Flex>
                </div>
                {node.oid === PENDING_FILE_OID && (
                  <ProgressBar
                    kind="indeterminate"
                    size="small"
                    aria-label="Uploading..."
                    className="mb-[-8px]"
                  />
                )}
              </Flex>
            ),
            onCellSelect: () => {
              if (node.type === 'file') {
                onFileSelect?.(node.path);
              } else if (node.type === 'directory') {
                handleUserFolderToggle(node.path);
              }
            },
            attributes: {
              TableDataCell: {
                // Directories always toggle on click; files only do so when a
                // view handler is wired up. Skip the pointer cursor for files
                // without a handler so the row doesn't look interactive.
                className: node.type === 'directory' || onFileSelect ? 'cursor-pointer' : undefined,
              },
            },
          },
          {
            children:
              node.type === 'file' ? getHumanReadableFileSize(node.size) : getFolderSize(node),
          },
          ...(extraColumns ?? []).map((col) => ({
            children: col.cell(node),
          })),
          {
            children:
              node.oid === PENDING_FILE_OID ? null : node.type === 'file' ? (
                <FileQuickActions
                  file={node}
                  datasetId={datasetId}
                  currentFolder={currentFolder}
                  onViewFile={onFileSelect}
                  isReadWriteDataset={isReadWriteDataset}
                />
              ) : node.type === 'directory' ? (
                <DirectoryQuickActions
                  directory={node}
                  datasetId={datasetId}
                  currentFolder={currentFolder}
                  showExpandSubtree={!searchQuery && folderHasSubfolders(node.path)}
                  isSubtreeExpanded={isFolderSubtreeExpanded(node.path)}
                  onToggleExpandSubtree={() => toggleFolderSubtreeExpand(node.path)}
                />
              ) : null,
            attributes: {
              TableDataCell: {
                style: { textOverflow: 'clip' },
                align: 'center',
                className: 'h-[59px]',
              },
            },
          },
        ],
      })),
    [
      treeRows,
      expandedFolders,
      handleUserFolderToggle,
      toggleFolderSubtreeExpand,
      isFolderSubtreeExpanded,
      folderHasSubfolders,
      datasetId,
      currentFolder,
      onFileSelect,
      isReadWriteDataset,
      selectedItems,
      addSelectedItem,
      removeSelectedItem,
      searchQuery,
      extraColumns,
    ]
  );
}
