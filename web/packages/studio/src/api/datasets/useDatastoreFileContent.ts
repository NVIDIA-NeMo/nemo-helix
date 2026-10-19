// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { queryOptions, useQuery, UseQueryOptions, useSuspenseQuery } from '@tanstack/react-query';

interface UseDatastoreFileContentParams {
  url: string;
  path: string;
  range?: [number, number];
}

export type UseDatastoreFilesOptions = Omit<
  UseQueryOptions<string, Error>,
  'queryFn' | 'queryKey'
> &
  UseDatastoreFileContentParams;

/**
 * Parses workspace and fileset name from an hf:// URL.
 * Format: hf://workspace/name or hf://datasets/workspace/name
 */
function parseHfUrl(url: string): { workspace: string; filesetName: string } {
  const cleanUrl = url.replace('hf://', '');
  const parts = cleanUrl.split('/');

  // Handle both hf://workspace/name and hf://datasets/workspace/name formats
  if (parts[0] === 'datasets' && parts.length >= 3) {
    return { workspace: parts[1], filesetName: parts[2] };
  }
  return { workspace: parts[0] || '', filesetName: parts[1] || '' };
}

const datastoreFileContentQueryOptions = ({ url, path, range }: UseDatastoreFilesOptions) => {
  const { workspace, filesetName } = parseHfUrl(url);

  return queryOptions<string, Error>({
    staleTime: Infinity,
    queryKey: ['datastore', 'file', url, path, range],
    queryFn: async () => {
      const blob = await filesDownloadFile(workspace, filesetName, path);
      if (!blob) throw new Error('Failed to download file');

      // Handle range requests
      if (range) {
        const slicedBlob = blob.slice(range[0], range[1]);
        return slicedBlob.text();
      }

      return blob.text();
    },
  });
};

export const useDatastoreFileContent = ({
  url,
  path,
  range,
  ...options
}: UseDatastoreFilesOptions) => {
  return useQuery({
    ...datastoreFileContentQueryOptions({ url, path, range }),
    ...options,
  });
};

export const useDatastoreFileContentSuspense = ({
  url,
  path,
  range,
  ...options
}: UseDatastoreFilesOptions) => {
  return useSuspenseQuery({
    ...datastoreFileContentQueryOptions({ url, path, range }),
    ...options,
  });
};
