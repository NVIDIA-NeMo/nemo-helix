// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileTag, type FileTagStatus } from '@nemo/common/src/components/FileTag';
import { type RenderFileTagFn, FileUpload } from '@nemo/common/src/components/FileUpload';
import type { ComponentProps, FC } from 'react';

export interface DatasetFileUploadProps extends Omit<
  ComponentProps<typeof FileUpload>,
  'onRemoveFile'
> {
  status?: FileTagStatus;
  onChange?: (files?: File[] | File) => void;
}

export const DatasetFileUpload: FC<DatasetFileUploadProps> = ({
  files,
  multiple = false,
  status,
  required = true,
  onChange,
  ...fileUploadProps
}) => {
  const handleFilesChange = (nextFiles: File[]) =>
    multiple ? onChange?.(nextFiles) : onChange?.(nextFiles[0]);

  const addFile = (newFiles: File[]) => {
    const existingFiles =
      files?.reduce(
        (cache, file) => {
          cache[file.name] = file.size;
          return cache;
        },
        {} as Record<string, number>
      ) ?? {};
    const nextFiles = newFiles
      .filter((file) => {
        return existingFiles[file.name] !== file.size;
      })
      .concat(files ?? []);
    handleFilesChange(nextFiles);
  };

  const removeFile = (file: File) => {
    handleFilesChange(
      files?.filter((checkFile) => {
        return checkFile.name !== file.name || checkFile.size !== file.size;
      }) ?? []
    );
  };

  const renderFileTag: RenderFileTagFn = (file, disabled, onRemove) => {
    return (
      <FileTag
        key={`file-tag-${file.name}`}
        fileName={file.name}
        status={status}
        disabled={disabled}
        onClick={onRemove}
        required={required}
      />
    );
  };

  return (
    <FileUpload
      required={required}
      status={status}
      files={files}
      renderFileTag={renderFileTag}
      multiple={multiple}
      onDropAccepted={addFile}
      {...fileUploadProps}
      onRemoveFile={removeFile}
    />
  );
};
