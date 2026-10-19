// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { FileListItem } from '@nemo/common/src/components/FileList';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

/**
 * Hook for file preview functionality.
 * Manages preview state and fetches file content when needed.
 */
export const useFilePreview = () => {
  const [previewFile, setPreviewFile] = useState<FileListItem | null>(null);

  const {
    data: previewContent,
    isLoading: isLoadingPreview,
    error: previewError,
  } = useQuery({
    queryKey: [
      'dataset-file-preview',
      previewFile?.dataset?.workspace ?? parseFilesetLocation(previewFile?.url ?? '')?.workspace,
      previewFile?.dataset?.name ?? parseFilesetLocation(previewFile?.url ?? '')?.name,
      previewFile?.path,
    ],
    queryFn: async () => {
      if (!previewFile) return null;

      // Use local content if available (e.g., for uploaded files)
      if (previewFile.content) {
        return previewFile.content;
      }

      // Try to get workspace/name from dataset, or fall back to parsing the fileset URL
      let workspace: string | undefined;
      let name: string | undefined;

      if (previewFile.dataset) {
        workspace = previewFile.dataset.workspace ?? undefined;
        name = previewFile.dataset.name ?? undefined;
      } else if (previewFile.url) {
        const parsed = parseFilesetLocation(previewFile.url);
        if (parsed) {
          workspace = parsed.workspace;
          name = parsed.name;
        }
      }

      if (!workspace || !name) {
        throw new Error('Missing dataset');
      }
      const response = await filesDownloadFile(workspace, name, previewFile.path);

      if (!response) throw new Error('Failed to fetch file content');
      return await response.text();
    },
    staleTime: Infinity,
    enabled: !!previewFile,
  });

  const clearPreview = () => {
    setPreviewFile(null);
  };

  return {
    previewFile,
    previewContent,
    isLoadingPreview,
    previewError: previewError as Error | null,
    setPreviewFile,
    clearPreview,
  };
};
