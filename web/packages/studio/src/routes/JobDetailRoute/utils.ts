// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { FileStorageType, type HelixJobResultResponse } from '@nemo/sdk/generated/platform/schema';

export interface ArtifactItem {
  resultName: string;
  workspace: string;
  fileset: string;
  objectPath: string;
}

export const resolveArtifactItems = (
  results: ReadonlyArray<HelixJobResultResponse>,
  workspaceFallback: string
): ArtifactItem[] => {
  const items: ArtifactItem[] = [];

  for (const result of results) {
    const url = result.artifact_url;
    if (!url) continue;
    if (result.artifact_storage_type !== FileStorageType.fileset) continue;
    if (!url.startsWith('fileset://') && url.includes(':')) continue;

    const parsed = parseFilesetLocation(url, workspaceFallback);
    if (!parsed?.objectPath) continue;

    items.push({
      resultName: result.name,
      workspace: parsed.workspace,
      fileset: parsed.name,
      objectPath: parsed.objectPath,
    });
  }

  return items;
};
