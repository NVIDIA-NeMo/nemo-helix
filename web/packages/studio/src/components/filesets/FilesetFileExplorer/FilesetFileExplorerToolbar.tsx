// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, TableToolbar, TextInput } from '@nvidia/foundations-react-core';
import { MagnifyingGlass } from '@nvidia/react-gui-icons/icons';
import { BulkActionsBar } from '@studio/components/filesets/FilesetFileExplorer/BulkActionsBar';
import type { FileSystemFile, FileSystemNode } from '@studio/components/FilesTable/utils';
import type { FC } from 'react';

export interface FilesetFileExplorerToolbarProps {
  selectedItems: FileSystemNode[];
  selectedFiles: FileSystemFile[];
  allSelectedAreFiles: boolean;
  isReadWriteDataset: boolean;
  workspace: string;
  datasetName: string;
  clearSelectedItems: () => void;
  isDuplicating: boolean;
  handleBulkDuplicate: (files: FileSystemFile[]) => Promise<boolean>;
  isDownloading: boolean;
  handleBulkDownload: (files: FileSystemFile[]) => Promise<void>;
  onMove: () => void;
  searchQuery: string;
  handleSearchQueryChange: (value: string, onClearSelection: () => void) => void;
  onNewDirectory: () => void;
  onUploadFile: () => void;
}

export const FilesetFileExplorerToolbar: FC<FilesetFileExplorerToolbarProps> = ({
  selectedItems,
  selectedFiles,
  allSelectedAreFiles,
  isReadWriteDataset,
  workspace,
  datasetName,
  clearSelectedItems,
  isDuplicating,
  handleBulkDuplicate,
  isDownloading,
  handleBulkDownload,
  onMove,
  searchQuery,
  handleSearchQueryChange,
  onNewDirectory,
  onUploadFile,
}) => (
  <TableToolbar
    aria-label="Dataset files toolbar"
    className="min-w-0 shrink-0"
    showBulkActionsToolbar={selectedItems.length > 0}
    slotBulkActions={
      <BulkActionsBar
        selectedItems={selectedItems}
        selectedFiles={selectedFiles}
        allSelectedAreFiles={allSelectedAreFiles}
        isReadWriteDataset={isReadWriteDataset}
        workspace={workspace}
        datasetName={datasetName}
        clearSelectedItems={clearSelectedItems}
        isDuplicating={isDuplicating}
        handleBulkDuplicate={handleBulkDuplicate}
        isDownloading={isDownloading}
        handleBulkDownload={handleBulkDownload}
        onMove={onMove}
      />
    }
  >
    <Flex direction="row" gap="density-md" className="min-w-0 w-full">
      <TextInput
        value={searchQuery}
        onValueChange={(value) => handleSearchQueryChange(value, clearSelectedItems)}
        placeholder="Search"
        slotStart={<MagnifyingGlass />}
        dismissible
        data-testid="dataset-details-search-input"
        className="min-w-0 flex-1"
      />
      {isReadWriteDataset && (
        <Flex gap="density-md" className="ml-auto">
          <Button
            kind="secondary"
            onClick={onNewDirectory}
            data-testid="dataset-details-new-directory-button"
          >
            New Directory
          </Button>
          <Button kind="secondary" onClick={onUploadFile}>
            Upload File
          </Button>
        </Flex>
      )}
    </Flex>
  </TableToolbar>
);
