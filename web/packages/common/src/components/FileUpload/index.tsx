// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileTag, FileTagStatus } from '@nemo/common/src/components/FileTag';
import {
  getAcceptedFileExtensions,
  hadInvalidFileTypeError,
  hadTooManyFilesError,
} from '@nemo/common/src/components/FileUpload/util';
import { InputErrorText } from '@nemo/common/src/components/InputErrorText';
import { Block, Flex, Label, Stack, Text } from '@nvidia/foundations-react-core';
import { DocumentNew, IconDocument } from '@nvidia/react-gui-icons/icons';
import { FC, MouseEvent, MouseEventHandler, ReactNode, useCallback, useState } from 'react';
import { DropEvent, DropzoneOptions, FileRejection, useDropzone } from 'react-dropzone';

const defaultRenderFileTag: RenderFileTagFn = (file, disabled, onRemove) => {
  return (
    <FileTag
      key={file.name + file.size}
      fileName={file.name}
      disabled={disabled}
      slotStart={<IconDocument />}
      onClick={onRemove}
      required
    />
  );
};

export type RenderFileTagFn = (
  file: File,
  disabled: boolean,
  onClick: MouseEventHandler<HTMLButtonElement>
) => ReactNode;

export interface FileUploadProps extends DropzoneOptions {
  label?: string;
  required?: boolean;
  helperText?: ReactNode;
  errorText?: ReactNode;
  disabled?: boolean;
  files?: File[];
  onRemoveFile: (file: File) => void;
  renderFileTag?: RenderFileTagFn;
  status?: FileTagStatus;
}

/**
 * A generic file upload that allows drag and drop
 */
export const FileUpload: FC<FileUploadProps> = ({
  label,
  helperText,
  errorText,
  disabled = false,
  files,
  onRemoveFile,
  renderFileTag = defaultRenderFileTag,
  accept,
  multiple = true,
  onDropAccepted,
  onDropRejected,
  status,
  ...dropOptions
}) => {
  const [fileErrorText, setFileErrorText] = useState<string>();

  const handleDropAccepted = useCallback(
    (acceptedFiles: File[], event: DropEvent) => {
      setFileErrorText(undefined);
      onDropAccepted?.(acceptedFiles, event);
    },
    [onDropAccepted]
  );

  const handleDropRejected = useCallback(
    (fileRejections: FileRejection[], event: DropEvent) => {
      if (hadTooManyFilesError(fileRejections)) {
        setFileErrorText('Only one file may be uploaded');
      } else if (hadInvalidFileTypeError(fileRejections)) {
        setFileErrorText(
          accept
            ? `File type must be one of: ${getAcceptedFileExtensions(accept).join(', ')}`
            : 'Invalid file type'
        );
        return;
      } else {
        setFileErrorText('An unknown error occurred');
      }
      onDropRejected?.(fileRejections, event);
    },
    [accept, onDropRejected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept,
    multiple,
    disabled,
    onDropAccepted: handleDropAccepted,
    onDropRejected: handleDropRejected,
    ...dropOptions,
    useFsAccessApi: false, // Fixes issue with react-dropzone and playwright.
  });

  const getDropBoxText = () => {
    if (isDragActive) {
      return 'Drop here...';
    }
    if (multiple) {
      return 'Drop files or click to select files';
    }
    return 'Drop a file or click to select a file';
  };

  const dropboxProps = {
    ...getRootProps(),
  };

  const createFileRemoveHandler = (file: File) => (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    setFileErrorText(undefined);
    onRemoveFile(file);
  };

  const getBorderColor = () => {
    if (status === 'error') return 'border-feedback-danger';
    if (status === 'success') return 'border-feedback-success';
    // 'pending' and 'idle' both use default border
    return 'border-input-border-default';
  };

  return (
    <Stack gap="density-xs">
      {label && <Label>{label}</Label>}
      <Block
        {...dropboxProps}
        className={`
          flex items-center gap-2 border-2 border-dashed ${getBorderColor()} bg-display-bg-canvas
          ${files ? 'p-1.5' : 'p-3'}
          ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}
        `}
      >
        <input {...getInputProps()} aria-label={label} data-testid="dropzone" />
        {files ? (
          <Stack gap="density-xs">
            {files.map((file) => renderFileTag(file, disabled, createFileRemoveHandler(file)))}
          </Stack>
        ) : (
          <Flex gap="density-xs" align="center" justify="center" className="w-full">
            <DocumentNew />
            <Text className="text-secondary">{getDropBoxText()}</Text>
          </Flex>
        )}
      </Block>
      {(fileErrorText || errorText) && (
        <InputErrorText>{fileErrorText || errorText}</InputErrorText>
      )}
      {helperText && <Text fontWeight="light">{helperText}</Text>}
    </Stack>
  );
};
