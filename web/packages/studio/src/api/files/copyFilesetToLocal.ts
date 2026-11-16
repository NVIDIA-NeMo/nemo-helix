// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { filesCreateFileset, filesDeleteFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { copyFilesetFiles } from '@studio/api/files/uploadFilesetEntries';

export interface CopyFilesetToLocalParams {
  readonly source: FilesetOutput;
  readonly name: string;
  readonly description?: string;
  readonly onProgress?: (copied: number, total: number) => void;
}

/** Creates a local-storage fileset in the source's workspace and copies every file into it,
 *  deleting the new fileset if any file fails to copy. If that cleanup also fails, the thrown
 *  error names the leftover fileset. */
export const copyFilesetToLocal = async ({
  source,
  name,
  description,
  onProgress,
}: CopyFilesetToLocalParams): Promise<FilesetOutput> => {
  const fileset = await filesCreateFileset(source.workspace, {
    name,
    description: description ?? source.description,
    purpose: source.purpose,
    metadata: source.metadata,
    custom_fields: source.custom_fields,
  });
  try {
    await copyFilesetFiles(source, fileset, onProgress);
  } catch (error) {
    const message = getErrorMessage(error as Error, 'Failed to copy fileset');
    await filesDeleteFileset(fileset.workspace, fileset.name).catch(() => {
      throw new Error(
        `${message}. The partial fileset ${fileset.workspace}/${fileset.name} could not be removed; delete it manually.`,
        { cause: error }
      );
    });
    throw error;
  }
  return fileset;
};
