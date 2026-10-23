// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CopyGeneric, Download, FolderOpen, Trash } from '@nemo/icons/icons';
import { Button, Flex, Text } from '@nvidia/foundations-react-core';
import { BulkDeleteModal } from '@studio/components/filesets/FilesetFileExplorer/BulkDeleteModal';
import type { FileSystemFile, FileSystemNode } from '@studio/components/FilesTable/utils';
import { getTextWithCount } from '@studio/util/strings';
import type { FC } from 'react';

export interface BulkActionsBarProps {
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
}

export const BulkActionsBar: FC<BulkActionsBarProps> = ({
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
}) => (
  <Flex
    align="center"
    justify="between"
    className="w-full"
    data-testid="dataset-files-selection-bar"
  >
    <Text kind="body/regular/md">
      {getTextWithCount('row', selectedItems.length, 'rows')} selected
    </Text>
    <Flex align="center" gap="density-md">
      {isReadWriteDataset ? (
        <BulkDeleteModal
          selectedItems={selectedItems}
          workspace={workspace}
          datasetName={datasetName}
          onConfirmDelete={clearSelectedItems}
          slotTrigger={
            <Button kind="tertiary" data-testid="dataset-files-bulk-delete">
              <Trash />
              Delete
            </Button>
          }
        />
      ) : null}
      {allSelectedAreFiles ? (
        <>
          {isReadWriteDataset ? (
            <>
              <Button
                kind="tertiary"
                disabled={isDuplicating}
                onClick={async () => {
                  // Keep the selection on failure so the user
                  // can retry without re-selecting.
                  const ok = await handleBulkDuplicate(selectedFiles);
                  if (ok) clearSelectedItems();
                }}
                data-testid="dataset-files-bulk-duplicate"
              >
                <CopyGeneric />
                Duplicate
              </Button>
              <Button kind="tertiary" onClick={onMove} data-testid="dataset-files-bulk-move">
                <FolderOpen />
                Move
              </Button>
            </>
          ) : null}
          <Button
            kind="tertiary"
            disabled={isDownloading}
            onClick={async () => {
              await handleBulkDownload(selectedFiles);
              clearSelectedItems();
            }}
            data-testid="dataset-files-bulk-download"
          >
            <Download />
            Download
          </Button>
        </>
      ) : null}
      <Button kind="tertiary" onClick={clearSelectedItems} data-testid="dataset-files-bulk-cancel">
        Cancel
      </Button>
    </Flex>
  </Flex>
);
