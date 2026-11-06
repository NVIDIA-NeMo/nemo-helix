// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FileContentsWithPath } from '@nemo/common/src/data-store/types';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { useDatasetFilesUpload } from '@studio/api/datasets/useDatasetFilesUpload';
import { renameFile } from '@studio/util/files';
import { useCallback, useState } from 'react';

export interface UseFileUploadOptions {
  workspace: string;
  datasetName: string;
  currentFolder?: string;
  filesList?: FilesetFileOutput[];
}

export interface UseFileUploadResult {
  /** @param targetFolder - when set, uploads into this folder instead of {@link UseFileUploadOptions.currentFolder} */
  handleUpload: (files: File[], targetFolder?: string) => Promise<void>;
  isUploading: boolean;
  pendingUploads?: (URL | File | FileContentsWithPath)[];
  pendingDuplicates: File[];
  confirmDuplicateUpload: () => Promise<void>;
  cancelDuplicateUpload: () => void;
}

export function useFileUpload(options: UseFileUploadOptions): UseFileUploadResult {
  const { workspace, datasetName, currentFolder, filesList } = options;
  const [pendingDuplicates, setPendingDuplicates] = useState<File[]>([]);

  const {
    mutateAsync: uploadFiles,
    variables: uploadVariables,
    isPending: isUploading,
  } = useDatasetFilesUpload();

  const handleUpload = useCallback(
    async (files: File[], targetFolder?: string) => {
      const folder = targetFolder ?? currentFolder;
      const prefixWithFolder = (file: File): File => {
        if (folder) {
          const prefix = folder.endsWith('/') ? folder : folder + '/';
          return renameFile(file, prefix + file.name);
        }
        return file;
      };
      const modifiedFiles = files.map(prefixWithFolder);
      const existingPaths = new Set(filesList?.map((f) => f.path) ?? []);

      const newFiles = modifiedFiles.filter((file) => !existingPaths.has(file.name));
      const duplicateFiles = modifiedFiles.filter((file) => existingPaths.has(file.name));

      if (newFiles.length > 0) {
        await uploadFiles({ workspace, datasetName, files: newFiles });
      }

      if (duplicateFiles.length > 0) {
        setPendingDuplicates(duplicateFiles);
      }
    },
    [uploadFiles, currentFolder, filesList, workspace, datasetName]
  );

  const confirmDuplicateUpload = useCallback(async () => {
    if (pendingDuplicates.length > 0) {
      await uploadFiles({ workspace, datasetName, files: pendingDuplicates });
      setPendingDuplicates([]);
    }
  }, [uploadFiles, pendingDuplicates, workspace, datasetName]);

  const cancelDuplicateUpload = useCallback(() => {
    setPendingDuplicates([]);
  }, []);

  return {
    handleUpload,
    isUploading,
    pendingUploads: uploadVariables?.files,
    pendingDuplicates,
    confirmDuplicateUpload,
    cancelDuplicateUpload,
  };
}
