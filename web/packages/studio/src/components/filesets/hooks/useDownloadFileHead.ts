// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@nemo/sdk/generated/fetchers/platform';
import { getFilesDownloadFileQueryKey } from '@nemo/sdk/generated/platform/files';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';

export interface DownloadFileHeadArgs {
  workspace: string;
  datasetName: string;
  path: string;
  /** Number of bytes to fetch. Defaults to 65536 (64 KB). */
  bytes?: number;
}

/**
 * Returns a callback that fetches the first N bytes of a fileset file via an
 * HTTP Range request. Results are cached by TanStack Query so repeated calls
 * for the same file (e.g. schema inference after a preview) return instantly.
 *
 * Auth is handled automatically by the Axios interceptor in customFetch.
 *
 * Resolves to `null` on any transport or HTTP error.
 */
export function useDownloadFileHead() {
  const queryClient = useQueryClient();

  return useCallback(
    async ({
      workspace,
      datasetName,
      path,
      bytes = 65536,
    }: DownloadFileHeadArgs): Promise<ArrayBuffer | null> => {
      try {
        const sdkQueryKey = getFilesDownloadFileQueryKey(
          encodeURIComponent(workspace),
          encodeURIComponent(datasetName),
          encodeURIComponent(path)
        );
        const blob = await queryClient.fetchQuery({
          queryKey: [...sdkQueryKey, 'range', bytes],
          staleTime: Infinity,
          queryFn: () =>
            customFetch<Blob>({
              url: sdkQueryKey[0],
              method: 'GET',
              responseType: 'blob',
              headers: { Range: `bytes=0-${bytes - 1}` },
            }),
        });
        return blob.arrayBuffer();
      } catch {
        return null;
      }
    },
    [queryClient]
  );
}
