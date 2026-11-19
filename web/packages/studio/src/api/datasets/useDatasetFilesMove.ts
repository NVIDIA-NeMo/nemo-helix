// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import type { EntityIdentifier } from '@studio/api/common/types';
import { invalidateDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { renameDatasetFile } from '@studio/api/datasets/useDatasetFileRename';
import { type UseMutationOptions, useMutation } from '@tanstack/react-query';

type MoveDatasetFilesParams = Required<EntityIdentifier> & {
  /** Array of file paths to move */
  filePaths: string[];
  /** Target folder path (empty string for root) */
  targetFolder: string;
};

export type UseDatasetFilesMoveOptions = Omit<
  UseMutationOptions<FilesetFileOutput[], Error, MoveDatasetFilesParams>,
  'mutationFn'
>;

/**
 * Gets the new file path after moving to a target folder.
 * Preserves the original filename but changes the directory.
 */
const getNewFilePath = (originalPath: string, targetFolder: string): string => {
  const fileName = originalPath.split('/').pop() || originalPath;

  if (!targetFolder) {
    // Moving to root
    return fileName;
  }

  // Ensure target folder doesn't have trailing slash
  const normalizedFolder = targetFolder.endsWith('/') ? targetFolder.slice(0, -1) : targetFolder;
  return `${normalizedFolder}/${fileName}`;
};

export const useDatasetFilesMove = (options?: UseDatasetFilesMoveOptions) => {
  const moveDatasetFiles = async ({
    workspace,
    name,
    filePaths,
    targetFolder,
  }: MoveDatasetFilesParams): Promise<FilesetFileOutput[]> => {
    const results: FilesetFileOutput[] = [];

    for (const path of filePaths) {
      const newFilePath = getNewFilePath(path, targetFolder);

      // Skip if file is already in target location
      if (path === newFilePath) {
        continue;
      }

      // Reuse the rename logic (download → upload → delete)
      const result = await renameDatasetFile({ workspace, name, path, newFilePath });
      results.push(result);
    }

    return results;
  };

  return useMutation({
    ...options,
    mutationFn: moveDatasetFiles,
    onSuccess: (data, variables, onMutateResult, context) => {
      invalidateDatasetCaches(variables.workspace, variables.name, ['files']);
      options?.onSuccess?.(data, variables, onMutateResult, context);
    },
    onError: (error, variables, onMutateResult, context) => {
      // Also refetch on error to show actual server state (some files may have moved)
      invalidateDatasetCaches(variables.workspace, variables.name, ['files']);
      options?.onError?.(error, variables, onMutateResult, context);
    },
  });
};
