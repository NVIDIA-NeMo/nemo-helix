// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { EntityIdentifier } from '@studio/api/common/types';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { GITKEEP_FILENAME } from '@studio/components/FilesTable/utils';
import { UseMutationOptions, useMutation } from '@tanstack/react-query';

export interface CreateDirectoryParams extends Required<EntityIdentifier> {
  /** Folder name to create (should not contain slashes) */
  folderName: string;
  /** Current folder path (undefined or empty for root) */
  currentFolder?: string;
}

/**
 * Creates a directory in a dataset by uploading a placeholder .gitkeep file.
 * Folders are created implicitly when uploading a file with a path.
 */
async function createDirectory({
  workspace,
  name,
  folderName,
  currentFolder,
}: CreateDirectoryParams): Promise<FilesetFileOutput> {
  const folderPath = currentFolder
    ? `${currentFolder}/${folderName}/${GITKEEP_FILENAME}`
    : `${folderName}/${GITKEEP_FILENAME}`;
  const emptyBlob = new Blob([], { type: 'application/octet-stream' });
  return filesUploadFile(workspace, name, folderPath, emptyBlob);
}

export type UseCreateDirectoryOptions = Omit<
  UseMutationOptions<FilesetFileOutput, Error, CreateDirectoryParams>,
  'mutationFn'
>;

export const useCreateDirectory = (options?: UseCreateDirectoryOptions) => {
  return useMutation({
    ...options,
    mutationFn: createDirectory,
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(variables.workspace, variables.name, ['files']);
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
