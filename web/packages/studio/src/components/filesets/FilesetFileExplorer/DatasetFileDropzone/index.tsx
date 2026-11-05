// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { Block, Flex, Text } from '@nvidia/foundations-react-core';
import { type FC, useCallback, type ReactNode } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';

type DatasetFileDropzoneProps = {
  children: (openFileDialog: () => void) => ReactNode;
  onUpload?: (files: File[]) => void;
  datasetName: string;
  /** When true, drag-drop intake and the picker are no-ops. Use for read-only
   *  filesets where uploads would be rejected by the backend anyway. */
  disabled?: boolean;
};

export const DatasetFileDropzone: FC<DatasetFileDropzoneProps> = ({
  children,
  onUpload,
  datasetName,
  disabled = false,
}) => {
  const handleDropAccepted = useCallback(
    (acceptedFiles: File[]) => {
      try {
        if (onUpload && acceptedFiles.length > 0) {
          onUpload(acceptedFiles);
        }
      } catch (error) {
        logger.error('Error handling accepted files', error);
      }
    },
    [onUpload]
  );

  const handleDropRejected = useCallback((fileRejections: FileRejection[]) => {
    logger.error(`Files rejected: ${JSON.stringify(fileRejections)}`);
  }, []);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    open: openFileDialog,
  } = useDropzone({
    onDropAccepted: handleDropAccepted,
    onDropRejected: handleDropRejected,
    multiple: true,
    useFsAccessApi: false, // Fixes issue with react-dropzone and playwright.
    noClick: true, // Prevent clicking on the dropzone from opening file dialog
    noKeyboard: true, // Prevent keyboard events from opening file dialog
    disabled,
  });

  return (
    <Block
      data-testid="dataset-file-dropzone"
      className="w-full h-full relative"
      padding="density-lg"
      {...getRootProps()}
    >
      <input
        {...getInputProps()}
        aria-label="Upload File"
        data-testid="dataset-file-dropzone-input"
      />

      {/* Children content */}
      {children(openFileDialog)}

      {/* Drag overlay */}
      {isDragActive && (
        <Block className="absolute inset-0 flex items-center justify-center bg-surface-raised/50 border-2 border-dashed border-brand p-8">
          <Flex gap="density-md" align="center" justify="center" direction="row">
            <Text kind="body/regular/md">
              Drop files into <Text kind="body/bold/md">{datasetName}</Text> — you will choose the
              destination folder next.
            </Text>
          </Flex>
        </Block>
      )}
    </Block>
  );
};
