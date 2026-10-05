// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { triggerDownload } from '@nemo/common/src/utils/file';
import {
  Button,
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  Flex,
} from '@nvidia/foundations-react-core';
import { useDatasetFileDelete } from '@studio/api/datasets/useDatasetFileDelete';
import { CreateFileSplitsModal } from '@studio/components/FilesTable/CreateFileSplitsModal';
import { RenameFileModal } from '@studio/components/FilesTable/RenameFileModal';
import { FileSystemFile, FileSystemNode } from '@studio/components/FilesTable/utils';
import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { useWorkers } from '@studio/providers/workers/useWorkers';
import LargeFileWorker from '@studio/workers/LargeFileWorker?worker';
import { Download as DownloadIcon, Pencil, Trash, Split, EllipsisVertical } from 'lucide-react';
import { FC, useState } from 'react';

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

  const { mutateAsync: deleteFile, error: deleteError } = useDatasetFileDelete();

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
      <DropdownRoot>
        <DropdownTrigger asChild showChevron={false}>
          <Button
            kind="tertiary"
            aria-label="Open file actions menu"
            className="absolute top-4 right-19"
          >
            <EllipsisVertical />
          </Button>
        </DropdownTrigger>
        <DropdownContent align="end">
          <DropdownItem onClick={openModalWithFile('createSplit')}>
            <Flex align="center" gap="density-sm">
              <Split size="20" />
              Create Split
            </Flex>
          </DropdownItem>
          <DropdownItem onClick={downloadFile}>
            <Flex align="center" gap="density-sm">
              <DownloadIcon size="20" />
              Download
            </Flex>
          </DropdownItem>
          <DropdownItem onClick={openModalWithFile('rename')}>
            <Flex align="center" gap="density-sm">
              <Pencil size="20" />
              Rename
            </Flex>
          </DropdownItem>
          <DropdownItem onClick={openModalWithFile('delete')} danger>
            <Flex align="center" gap="density-sm">
              <Trash size="20" />
              Delete
            </Flex>
          </DropdownItem>
        </DropdownContent>
      </DropdownRoot>
      {openModal === 'delete' && modalFile && (
        <DeleteConfirmationModal
          open
          onDelete={handleDeleteFile}
          simpleConfirm
          title="Delete File"
          confirmationText={file.path}
          errorText={deleteError?.message}
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
