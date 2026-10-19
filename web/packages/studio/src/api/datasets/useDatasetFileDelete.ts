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

export interface DeleteDatasetFileParams {
  workspace: string;
  datasetName: string;
  path: string;
}

export type UseDatasetFileDeleteOptions = Omit<
  UseMutationOptions<FilesetFileOutput, Error, DeleteDatasetFileParams>,
  'mutationFn'
>;

export const useDatasetFileDelete = (options?: UseDatasetFileDeleteOptions) => {
  return useMutation({
    ...options,
    mutationFn: ({ workspace, datasetName, path }) => filesDeleteFile(workspace, datasetName, path),
    onSuccess: (data, variables, onMutateResult, context) => {
      queryClient.setQueriesData(
        { queryKey: getFilesListFilesetFilesQueryKey(variables.workspace, variables.datasetName) },
        (prev: ListFilesetFilesResponse | undefined) => {
          if (!prev) return prev;
          return {
            ...prev,
            data: prev.data.filter((file) => file.path !== variables.path),
          };
        }
      );
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
  });
};
