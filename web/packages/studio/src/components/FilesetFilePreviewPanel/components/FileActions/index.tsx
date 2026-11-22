// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { triggerDownload } from '@nemo/common/src/utils/file';
import { useDatasetFileDelete } from '@studio/api/datasets/useDatasetFileDelete';
import { CreateFileSplitsModal } from '@studio/components/FilesTable/CreateFileSplitsModal';
import { RenameFileModal } from '@studio/components/FilesTable/RenameFileModal';
import type { FileSystemFile, FileSystemNode } from '@studio/components/FilesTable/utils';
import { getDeleteFilesDescription } from '@studio/constants/filesets';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { useWorkers } from '@studio/providers/workers/useWorkers';
import LargeFileWorker from '@studio/workers/LargeFileWorker?worker';
import { type FC, useState } from 'react';

type ModalType = 'createSplit' | 'rename' | 'delete';

interface Props {
  /** Fileset workspace (e.g., 'my-workspace') */
  workspace: string;
  /** Fileset name (e.g., 'my-dataset' / 'my-model') */
  filesetName: string;
  /** File to perform actions on */
  file: FileSystemFile;
  /** Callback when file is successfully deleted */
  onDeleteSuccess?: () => void;
  /** Callback when file is successfully renamed */
  onRenameSuccess?: (newPath: string) => void;
}

export const FileActions: FC<Props> = ({
  workspace,
  filesetName,
  file,
  onDeleteSuccess,
  onRenameSuccess,
}) => {
  const [modalFile, setModalFile] = useState<FileSystemNode | undefined>();
  const [openModal, setOpenModal] = useState<ModalType | undefined>();
  const toast = useToast();
  const bearerToken = useOidcBearerToken();
  const { createWorker } = useWorkers();

  const { mutateAsync: deleteFile } = useDatasetFileDelete();

  const downloadFile = async () => {
    const worker = new LargeFileWorker();
    createWorker(worker, {
      onMessage: (e) => {
        const { done, arrayBuffer, error } = e.data;
        if (done && arrayBuffer) {
          triggerDownload(arrayBuffer, file.path);
          toast.success('Successfully downloaded file!');
        } else if (done && error) {
          toast.error(`Download failed: ${error}`);
        }
      },
      onError: () => {
        toast.error('Unable to download file. Please try again later.');
      },
    });
    worker.postMessage({
      action: 'downloadAsFile',
      workspace,
      dataset: filesetName,
      path: file.path,
      accessToken: bearerToken,
    });
  };

  const handleDeleteFile = async () => {
    if (!workspace || !filesetName) {
      toast.error('Failed to delete file: invalid fileset name');
      return false;
    }

    try {
      const response = await deleteFile({
        workspace,
        datasetName: filesetName,
        path: file.path,
      });
      if (response) {
        onDeleteSuccess?.();
      }
      return Boolean(response);
    } catch {
      return false;
    }
  };

  const handleRenameSuccess = (newPath: string) => {
    onRenameSuccess?.(newPath);
  };

  const openModalWithFile = (modal: ModalType) => () => {
    setModalFile(file);
    setOpenModal(modal);
  };

  return (
    <>
      <div className="absolute top-4 right-19">
        <QuickActionsMenuRoot
          ariaLabel="Open file actions menu"
          actions={[
            { label: 'Create Split', onSelect: openModalWithFile('createSplit') },
            { label: 'Download', onSelect: downloadFile },
            { label: 'Rename', onSelect: openModalWithFile('rename') },
          ]}
          deleteAction={{ onSelect: openModalWithFile('delete') }}
        />
      </div>
      {openModal === 'delete' && modalFile && (
        <DeleteConfirmationModal
          open
          entity="file"
          ids={[file.path]}
          description={getDeleteFilesDescription(1)}
          onDelete={handleDeleteFile}
          onClose={() => setOpenModal(undefined)}
        />
      )}
      {openModal === 'rename' && modalFile && (
        <RenameFileModal
          open
          filepath={file.path}
          onClose={() => setOpenModal(undefined)}
          onSuccess={handleRenameSuccess}
        />
      )}
      {openModal === 'createSplit' && modalFile && (
        <CreateFileSplitsModal open onClose={() => setOpenModal(undefined)} filepath={file.path} />
      )}
    </>
  );
};
