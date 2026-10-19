// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDeleteFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';

export interface DeleteDatasetDirectoryParams {
  workspace: string;
  datasetName: string;
  path: string;
}

export type UseDatasetDirectoryDeleteOptions = Omit<
  UseMutationOptions<FilesetFileOutput[], Error, DeleteDatasetDirectoryParams>,
  'mutationFn'
>;

/**
 * Deletes all files in a directory from a fileset.
 */
async function deleteDirectoryFromDataset({
  workspace,
  datasetName,
  path,
}: DeleteDatasetDirectoryParams): Promise<FilesetFileOutput[]> {
  // Get all files in the directory
  const response = await filesListFilesetFiles(workspace, datasetName, { path });
  const filePaths = response.data.map((file) => file.path);

  // Delete all files
  const results = await Promise.all(
    filePaths.map((filePath) => filesDeleteFile(workspace, datasetName, filePath))
  );
  return results;
}

export const useDatasetDirectoryDelete = (options?: UseDatasetDirectoryDeleteOptions) => {
  return useMutation({
    ...options,
    mutationFn: deleteDirectoryFromDataset,
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(variables.workspace, variables.datasetName, ['files']);
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
