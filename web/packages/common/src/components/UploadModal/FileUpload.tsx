// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useUploadModalContext } from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import { formatFileSize } from '@nemo/common/src/components/UploadModal/utils';
import { Upload, Text, Stack } from '@nvidia/foundations-react-core';
import { FC } from 'react';

export interface FileUploadProps {
  /** Callback when a file is selected */
  onFileAccepted?: (files: File[]) => void;
  /** Array of accepted file extensions (e.g., ['.json', '.jsonl']) */
  acceptedFileTypes?: string[];
  /** Maximum file size label to display */
  maxFileSizeLabel?: string;
  /** Whether the upload is disabled */
  disabled?: boolean;
  error?: string;
}

type FileUploadItem = {
  id: string;
  file: File;
  errorMessage?: string;
  status: 'error' | 'success' | 'uploading';
  uploadedBytes?: number;
  hidePreview?: boolean;
};
/**
 * A simple file upload component that wraps KUI Upload for single file uploads
 */
export const FileUpload: FC<FileUploadProps> = ({
  acceptedFileTypes,
  disabled = false,
  error,
  onFileAccepted,
}) => {
  const [state, dispatch] = useUploadModalContext();
  const {
    acceptableFileTypes: acceptableFileTypesState,
    acceptableFileSize,
    allowMultipleFileSelection,
  } = state;
  const handleFileChange = (fileOrFiles: FileUploadItem | FileUploadItem[]) => {
    const files = Array.isArray(fileOrFiles) ? fileOrFiles : [fileOrFiles];
    dispatch({
      type: 'SET_FILES',
      payload: files.map(({ file }) => ({
        id: `${file.name}-${file.size}-${file.lastModified}`,
        type: 'new',
        file,
      })),
    });
    onFileAccepted?.(files.map((file) => file.file));
  };

  return (
    <Stack className="w-full" gap="density-xs">
      <Upload
        className="w-full"
        accept={acceptedFileTypes?.join(',') ?? acceptableFileTypesState.join(',')}
        disabled={disabled}
        multiple={allowMultipleFileSelection}
        onValueChange={handleFileChange}
      >
        Up to {formatFileSize(acceptableFileSize ?? 0)}
      </Upload>
      {error && (
        <Text kind="label/regular/sm" className="text-feedback-danger">
          {error}
        </Text>
      )}
    </Stack>
  );
};
