// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  filesDeleteFile,
  filesDownloadFile,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { EntityIdentifier } from '@studio/api/common/types';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { UseMutationOptions, useMutation } from '@tanstack/react-query';

type RenameDatasetFileParams = Required<EntityIdentifier> & {
  path: string;
  newFilePath: string;
};

export type UseDatasetFileRenameOptions = Omit<
  UseMutationOptions<FilesetFileOutput, Error, RenameDatasetFileParams>,
  'mutationFn'
>;

/**
 * Core rename logic: download file, upload to new path, delete original.
 * Exported for reuse in other mutations (e.g., move files).
 */
export const renameDatasetFile = async ({
  workspace,
  name,
  path,
  newFilePath,
}: RenameDatasetFileParams): Promise<FilesetFileOutput> => {
  // Download the file
  const blob = await filesDownloadFile(workspace!, name!, path);
  if (!blob) {
    throw new Error('Invalid response while downloading file.');
  }

  // Upload to new path
  const uploadResult = await filesUploadFile(workspace!, name!, newFilePath, blob);

  // Delete the original
  await filesDeleteFile(workspace!, name!, path);

  return uploadResult;
};

export const useDatasetFileRename = (options?: UseDatasetFileRenameOptions) => {
  return useMutation({
    ...options,
    mutationFn: renameDatasetFile,
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(variables.workspace, variables.name, ['files']);
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
