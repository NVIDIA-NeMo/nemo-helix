// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  filesDeleteFile,
  getFilesListFilesetFilesQueryKey,
} from '@nemo/sdk/generated/platform/files';
import type {
  FilesetFileOutput,
  ListFilesetFilesResponse,
} from '@nemo/sdk/generated/platform/schema';
import { queryClient } from '@studio/api/queryClient';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';

export interface DeleteDatasetFilesParams {
  workspace: string;
  datasetName: string;
  paths: string[];
}

export type UseDatasetFilesDeleteOptions = Omit<
  UseMutationOptions<FilesetFileOutput[], Error, DeleteDatasetFilesParams>,
  'mutationFn'
>;

/**
 * Deletes multiple files from a fileset.
 */
async function deleteFilesFromDataset({
  workspace,
  datasetName,
  paths,
}: DeleteDatasetFilesParams): Promise<FilesetFileOutput[]> {
  const results = await Promise.all(
    paths.map((path) => filesDeleteFile(workspace, datasetName, path))
  );
  return results;
}

export const useDatasetFilesDelete = (options?: UseDatasetFilesDeleteOptions) => {
  return useMutation({
    ...options,
    mutationFn: deleteFilesFromDataset,
    onSuccess: (data, variables, onMutateResult, context) => {
      queryClient.setQueriesData(
        { queryKey: getFilesListFilesetFilesQueryKey(variables.workspace, variables.datasetName) },
        (prev: ListFilesetFilesResponse | undefined) => {
          if (!prev) return prev;
          return {
            ...prev,
            data: prev.data.filter((file) => !variables.paths.includes(file.path)),
          };
        }
      );
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
