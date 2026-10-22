// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  Button,
  Flex,
  Modal,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
  Upload,
  type UploadRootProps,
} from '@nvidia/foundations-react-core';
import { FolderClosed, Upload as UploadIcon } from '@nvidia/react-gui-icons/icons';
import { collectFolderPathsFromDatasetFiles } from '@studio/util/files';
import { FC, useEffect, useMemo, useRef, useState } from 'react';

// `FileUploadItem` isn't re-exported from the package root, so derive it.
type FileUploadItem = NonNullable<Extract<UploadRootProps, { multiple: true }>['value']>[number];

const ROOT_VALUE = '__root__';

const normalizeFolderOption = (folder: string | undefined): string => {
  if (!folder?.trim()) return ROOT_VALUE;
  const t = folder.trim();
  return t.endsWith('/') ? t.slice(0, -1) : t;
};

const toUploadItems = (files: File[]): FileUploadItem[] =>
  files.map((file, index) => ({
    id: `${file.name}-${file.lastModified}-${file.size}-${index}`,
    file,
    status: 'success',
  }));

export interface UploadToFolderModalProps {
  open: boolean;
  onClose: () => void;
  /** Files staged before the modal opened (e.g. dropped onto the explorer) */
  files: File[];
  /** Default folder path from breadcrumbs (without trailing slash) */
  defaultFolder?: string;
  filesList: FilesetFileOutput[] | undefined;
  /** Called with files and folder path (undefined = dataset root) */
  onConfirm: (files: File[], destinationFolder: string | undefined) => void | Promise<void>;
}

/**
 * Lets the user choose files and which folder they are uploaded into before starting the upload.
 */
export const UploadToFolderModal: FC<UploadToFolderModalProps> = ({
  open,
  onClose,
  files,
  defaultFolder,
  filesList,
  onConfirm,
}) => {
  const folderOptions = useMemo(() => collectFolderPathsFromDatasetFiles(filesList), [filesList]);
  const [selectedFolder, setSelectedFolder] = useState<string>(ROOT_VALUE);
  const [uploadItems, setUploadItems] = useState<FileUploadItem[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submitLockRef = useRef(false);

  useEffect(() => {
    if (open) {
      setSelectedFolder(normalizeFolderOption(defaultFolder));
      setUploadItems(toUploadItems(files));
    } else {
      setUploadItems([]);
      setIsSubmitting(false);
      submitLockRef.current = false;
    }
  }, [open, defaultFolder, files]);

  const destinationLabel = selectedFolder === ROOT_VALUE ? 'dataset root' : `${selectedFolder}/`;

  const handleConfirm = async () => {
    if (uploadItems.length === 0 || submitLockRef.current) return;
    const folder = selectedFolder === ROOT_VALUE ? undefined : selectedFolder;
    submitLockRef.current = true;
    setIsSubmitting(true);
    try {
      await onConfirm(
        uploadItems.map((item) => item.file),
        folder
      );
    } finally {
      submitLockRef.current = false;
      setIsSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) onClose();
      }}
      slotHeading={
        <>
          <UploadIcon />
          Upload files
        </>
      }
      slotFooter={
        <Flex justify="end" gap="density-xs" align="center" className="w-full">
          <Button kind="tertiary" color="neutral" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <LoadingButton
            onClick={handleConfirm}
            disabled={uploadItems.length === 0}
            loading={isSubmitting}
          >
            Upload
          </LoadingButton>
        </Flex>
      }
    >
      <Stack gap="density-md">
        <Text kind="body/regular/md">
          Choose where to upload{' '}
          {uploadItems.length > 0
            ? `${uploadItems.length} file${uploadItems.length !== 1 ? 's' : ''}`
            : 'your files'}{' '}
          (destination: {destinationLabel}).
        </Text>
        <Flex direction="col" gap="density-xs" align="stretch">
          <Text kind="label/bold/sm">Destination folder</Text>
          <SelectRoot
            value={selectedFolder}
            onValueChange={setSelectedFolder}
            disabled={isSubmitting}
          >
            <SelectTrigger
              placeholder="Select folder"
              aria-label="Upload destination folder"
              slotStart={<FolderClosed className="size-4 shrink-0" />}
            />
            <SelectContent>
              <SelectListbox>
                <SelectItem value={ROOT_VALUE}>Root</SelectItem>
                {folderOptions.map((path) => (
                  <SelectItem key={path} value={path}>
                    {path}
                  </SelectItem>
                ))}
              </SelectListbox>
            </SelectContent>
          </SelectRoot>
        </Flex>
        <Upload
          multiple
          disabled={isSubmitting}
          value={uploadItems}
          onValueChange={setUploadItems}
          attributes={{ UploadInputElement: { 'aria-label': 'Upload files' } }}
        />
      </Stack>
    </Modal>
  );
};
