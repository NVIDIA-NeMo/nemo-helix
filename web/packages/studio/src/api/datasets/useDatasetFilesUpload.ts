// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { UseMutationOptions, useMutation } from '@tanstack/react-query';

export interface UploadDatasetFilesParams {
  workspace: string;
  datasetName: string;
  files: File[];
}

export type UseDatasetFilesUploadOptions = Omit<
  UseMutationOptions<FilesetFileOutput[], Error, UploadDatasetFilesParams>,
  'mutationFn'
>;

/**
 * Uploads multiple files to a fileset.
 */
async function uploadFilesToDataset({
  workspace,
  datasetName,
  files,
}: UploadDatasetFilesParams): Promise<FilesetFileOutput[]> {
  const results = await Promise.all(
    files.map(async (file) => {
      const blob = new Blob([await file.arrayBuffer()], {
        type: file.type || 'application/octet-stream',
      });
      return filesUploadFile(workspace, datasetName, file.name, blob);
    })
  );
  return results;
}

export const useDatasetFilesUpload = (options?: UseDatasetFilesUploadOptions) => {
  return useMutation({
    ...options,
    mutationFn: uploadFilesToDataset,
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(variables.workspace, variables.datasetName, ['files', 'content']);
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
