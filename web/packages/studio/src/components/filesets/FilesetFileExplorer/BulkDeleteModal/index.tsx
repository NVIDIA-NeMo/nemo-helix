// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button } from '@nvidia/foundations-react-core';
import { Trash } from '@nvidia/react-gui-icons/icons';
import { useDatasetFilesDelete } from '@studio/api/datasets/useDatasetFilesDelete';
import { BulkDeleteModal as GenericBulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { extractFilePathsFromDirectory } from '@studio/components/filesets/FilesetFileExplorer/extractFilePathsFromDirectory';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import {
  cloneElement,
  isValidElement,
  type FC,
  type MouseEventHandler,
  type ReactNode,
  useState,
} from 'react';

interface TriggerProps {
  onClick?: MouseEventHandler;
}

interface BulkDeleteModalProps {
  selectedItems: FileSystemNode[];
  workspace: string;
  datasetName: string;
  onConfirmDelete: () => void;
  /** Custom trigger; defaults to a secondary Button */
  slotTrigger?: ReactNode;
}

export const BulkDeleteModal: FC<BulkDeleteModalProps> = ({
  selectedItems,
  workspace,
  datasetName,
  onConfirmDelete,
  slotTrigger,
}) => {
  const [open, setOpen] = useState(false);
  const { mutateAsync: deleteFiles } = useDatasetFilesDelete();

  const handleDelete = async (items: FileSystemNode[]) => {
    const directFilePaths = items.filter((i) => i.type === 'file').map((i) => i.path);
    const directoryFilePaths = items
      .filter((i) => i.type === 'directory')
      .flatMap((d) => extractFilePathsFromDirectory(d));
    const allFilePaths = [...directFilePaths, ...directoryFilePaths];
    if (allFilePaths.length > 0) {
      await deleteFiles({ workspace, datasetName, paths: allFilePaths });
    }
    onConfirmDelete();
  };

  const openTrigger = () => setOpen(true);

  const trigger = isValidElement<TriggerProps>(slotTrigger) ? (
    cloneElement(slotTrigger, {
      onClick: (e: Parameters<MouseEventHandler>[0]) => {
        slotTrigger.props.onClick?.(e);
        openTrigger();
      },
    })
  ) : (
    <Button kind="secondary" data-testid="bulk-delete-modal-trigger-button" onClick={openTrigger}>
      <Trash />
      Delete
    </Button>
  );

  return (
    <>
      {trigger}
      <GenericBulkDeleteModal
        items={selectedItems}
        open={open}
        onDelete={handleDelete}
        title={(count) => `Delete ${count} Item${count !== 1 ? 's' : ''}`}
        onClose={() => setOpen(false)}
      />
    </>
  );
};
