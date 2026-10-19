// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesCreateFileset, filesUploadFile } from '@nemo/sdk/generated/platform/files';
import type { CreateFilesetRequest, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { resetDatasetCaches } from '@studio/api/datasets/invalidateDatasetCaches';
import { useMutation, UseMutationOptions } from '@tanstack/react-query';

export interface CreateDatasetWithFilesParams {
  /** Workspace for the fileset */
  workspace?: string;
  /** Request to create a new fileset. If `dataset` is provided, this is ignored. */
  request?: Omit<CreateFilesetRequest, 'storage'>;
  /** Existing fileset to upload files to (for retry scenarios) */
  dataset?: FilesetOutput;
  files?: File[];
}

export type UseDatasetCreateOptions = Omit<
  UseMutationOptions<FilesetOutput, Error, CreateDatasetWithFilesParams>,
  'mutationFn'
>;

/**
 * Creates a fileset and optionally uploads files to it.
 * If `dataset` is provided, skips creation and only uploads files (for retry scenarios).
 */
async function createDatasetWithFiles({
  workspace,
  request,
  dataset,
  files,
}: CreateDatasetWithFilesParams): Promise<FilesetOutput> {
  // If an existing dataset is provided, use it (retry scenario)
  // Otherwise, create a new fileset
  let fileset: FilesetOutput;
  if (dataset) {
    fileset = dataset;
  } else {
    if (!workspace || !request) {
      throw new Error('workspace and request are required when creating a new fileset');
    }
    // Historically this flow only created dataset-purpose filesets, so 'dataset' is
    // used as the fallback when the caller doesn't pass a purpose. Callers that
    // support the Purpose selector (e.g. FilesetNewRoute) pass request.purpose
    // explicitly and it will take precedence here.
    fileset = await filesCreateFileset(workspace, { purpose: 'dataset', ...request });
  }

  const targetWorkspace = fileset.workspace;

  if (files?.length) {
    await Promise.all(
      files.map(async (file) => {
        const blob = new Blob([await file.arrayBuffer()], {
          type: file.type || 'application/octet-stream',
        });
        return filesUploadFile(targetWorkspace, fileset.name, file.name, blob);
      })
    );
  }

  return fileset;
}

export const useDatasetCreate = (options?: UseDatasetCreateOptions) => {
  return useMutation({
    ...options,
    mutationFn: createDatasetWithFiles,
    onSuccess: (...args) => {
      const [fileset] = args;
      resetDatasetCaches(fileset.workspace, fileset.name);
      options?.onSuccess?.(...args);
    },
  });
};
