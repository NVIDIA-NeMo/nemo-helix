// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { Trash } from '@nemo/icons/icons';
import { Button } from '@nvidia/foundations-react-core';
import { useDatasetFilesDelete } from '@studio/api/datasets/useDatasetFilesDelete';
import { extractFilePathsFromDirectory } from '@studio/components/filesets/FilesetFileExplorer/extractFilePathsFromDirectory';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import {
  DELETE_FILES_AND_FOLDERS_DESCRIPTION,
  getDeleteFilesDescription,
  getDeleteFoldersDescription,
} from '@studio/constants/filesets';
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

const getDeleteTarget = (items: FileSystemNode[]) => {
  if (items.every((item) => item.type === 'file')) {
    return { entity: 'file', description: getDeleteFilesDescription(items.length) };
  }
  if (items.every((item) => item.type === 'directory')) {
    return { entity: 'folder', description: getDeleteFoldersDescription(items.length) };
  }
  return { entity: 'item', description: DELETE_FILES_AND_FOLDERS_DESCRIPTION };
};

export const BulkDeleteModal: FC<BulkDeleteModalProps> = ({
  selectedItems,
  workspace,
  datasetName,
  onConfirmDelete,
  slotTrigger,
}) => {
  const [itemsToDelete, setItemsToDelete] = useState<FileSystemNode[]>([]);
  const { mutateAsync: deleteFiles } = useDatasetFilesDelete();

  const handleDelete = async () => {
    const directFilePaths = itemsToDelete.filter((i) => i.type === 'file').map((i) => i.path);
    const directoryFilePaths = itemsToDelete
      .filter((i) => i.type === 'directory')
      .flatMap((d) => extractFilePathsFromDirectory(d));
    const allFilePaths = [...directFilePaths, ...directoryFilePaths];
    if (allFilePaths.length > 0) {
      await deleteFiles({ workspace, datasetName, paths: allFilePaths });
    }
    onConfirmDelete();
    return true;
  };

  const openTrigger = () => setItemsToDelete(selectedItems);

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
      {itemsToDelete.length > 0 && (
        <DeleteConfirmationModal
          open
          {...getDeleteTarget(itemsToDelete)}
          ids={itemsToDelete.map((item) => item.path)}
          onDelete={handleDelete}
          onClose={() => setItemsToDelete([])}
        />
      )}
    </>
  );
};
