// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  type QuickActionItem,
  QuickActionsMenuRoot,
} from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useDatasetFileDelete } from '@studio/api/datasets/useDatasetFileDelete';
import { AddToFolderModal } from '@studio/components/filesets/AddToFolderModal';
import { useBulkDownload } from '@studio/components/filesets/hooks/useBulkDownload';
import { useBulkDuplicate } from '@studio/components/filesets/hooks/useBulkDuplicate';
import { CreateFileSplitsModal } from '@studio/components/FilesTable/CreateFileSplitsModal';
import { RenameFileModal } from '@studio/components/FilesTable/RenameFileModal';
import { TransformFileModal } from '@studio/components/FilesTable/TransformFileModal';
import type { FileSystemFile, FileSystemNode } from '@studio/components/FilesTable/utils';
import { useSelectedDatasetId } from '@studio/hooks/useSelectedDatasetId';
import { resolveDatasetFilePath } from '@studio/util/files';
import { type FC, useState } from 'react';

type ModalType = 'createSplit' | 'rename' | 'delete' | 'info' | 'transform' | 'addToFolder';

interface Props {
  /** Dataset ID (workspace/name or with folder path) */
  datasetId?: string;
  /** File to perform actions on */
  file: FileSystemFile;
  /** Current folder path (optional, for constructing full file paths) */
  currentFolder?: string;
  /** Callback when user wants to view/open a file (e.g., "View File" action) */
  onViewFile?: (filePath: string) => void;
  /** When true, show full menu (Move, Duplicate, Create Split, Transform, Rename). Read/write storage: local, s3. Read-only: ngc, huggingface. */
  isReadWriteDataset?: boolean;
}
export const FileQuickActions: FC<Props> = ({
  datasetId,
  file,
  currentFolder,
  onViewFile,
  isReadWriteDataset = false,
}) => {
  const [modalFile, setModalFile] = useState<FileSystemNode | undefined>();
  const [openModal, setOpenModal] = useState<ModalType | undefined>();
  const toast = useToast();
  const datasetFullName = useSelectedDatasetId({ datasetId });
  const { workspace, name } = getPartsFromReference(datasetFullName);

  const { mutateAsync, error: deleteError } = useDatasetFileDelete();

  // Use currentFolder only for basename-only paths; tree/API paths are already full paths
  const path = resolveDatasetFilePath(file.path, currentFolder ?? undefined);

  const resolvedFile: FileSystemFile = { ...file, path };

  const { handleBulkDownload, isDownloading } = useBulkDownload({
    workspace,
    datasetName: name,
  });
  const { handleBulkDuplicate, isDuplicating } = useBulkDuplicate({
    workspace,
    datasetName: name,
  });

  const downloadFile = () => {
    if (!workspace || !name) {
      toast.error('Failed to download: invalid dataset');
      return;
    }
    return handleBulkDownload([resolvedFile]);
  };
  const handleDuplicate = () => {
    if (!workspace || !name) {
      toast.error('Failed to duplicate: invalid dataset');
      return;
    }
    return handleBulkDuplicate([resolvedFile]);
  };

  const handleDeleteFile = async () => {
    if (!workspace || !name) {
      toast.error('Failed to delete file: invalid dataset name');
      return false;
    }

    try {
      const response = await mutateAsync({ workspace, datasetName: name, path });
      return Boolean(response);
    } catch {
      return false;
    }
  };

  const handleViewFile = () => {
    onViewFile?.(path);
  };

  const openModalWithFile = (modal: ModalType) => () => {
    setModalFile(file);
    setOpenModal(modal);
  };

  const handleCopyPath = async () => {
    try {
      await navigator.clipboard.writeText(path);
      toast.success('Path copied to clipboard');
    } catch {
      toast.error('Failed to copy path');
    }
  };

  const actions: QuickActionItem[] = isReadWriteDataset
    ? [
        ...(onViewFile ? [{ label: 'View File', onSelect: handleViewFile }] : []),
        { label: 'Download', onSelect: downloadFile, disabled: isDownloading },
        { label: 'Copy Path', onSelect: handleCopyPath, divider: { width: 'large' } },
        { label: 'Move', onSelect: openModalWithFile('addToFolder') },
        { label: 'Duplicate', onSelect: handleDuplicate, disabled: isDuplicating },
        { label: 'Create Split', onSelect: openModalWithFile('createSplit') },
        { label: 'Transform', onSelect: openModalWithFile('transform') },
        { label: 'Rename', onSelect: openModalWithFile('rename'), divider: { width: 'large' } },
        { label: 'Delete', onSelect: openModalWithFile('delete'), danger: true },
      ]
    : [
        ...(onViewFile ? [{ label: 'View File', onSelect: handleViewFile }] : []),
        { label: 'Download', onSelect: downloadFile, disabled: isDownloading },
        { label: 'Copy Path', onSelect: handleCopyPath },
      ];

  return (
    <>
      <QuickActionsMenuRoot actions={actions} />
      {openModal === 'delete' && modalFile && (
        <DeleteConfirmationModal
          open
          onDelete={handleDeleteFile}
          simpleConfirm
          title="Delete File"
          confirmationText={path}
          errorText={deleteError?.message}
          onClose={() => setOpenModal(undefined)}
        />
      )}
      {openModal === 'rename' && modalFile && (
        <RenameFileModal
          open
          onClose={() => setOpenModal(undefined)}
          filepath={path}
          datasetId={datasetFullName}
        />
      )}
      {openModal === 'createSplit' && modalFile && (
        <CreateFileSplitsModal
          open
          onClose={() => setOpenModal(undefined)}
          filepath={path}
          datasetId={datasetFullName}
        />
      )}
      {openModal === 'transform' && modalFile && (
        <TransformFileModal
          open
          onClose={() => setOpenModal(undefined)}
          filepath={path}
          datasetId={datasetFullName}
        />
      )}
      {openModal === 'addToFolder' && modalFile && workspace && name && (
        <AddToFolderModal
          open
          onClose={() => setOpenModal(undefined)}
          selectedItems={[file]}
          workspace={workspace}
          datasetName={name}
          currentFolder={currentFolder}
        />
      )}
    </>
  );
};
