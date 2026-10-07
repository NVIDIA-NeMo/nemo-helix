// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useOidcBearerToken } from '@studio/providers/auth/useOidcBearerToken';
import { useWorkers } from '@studio/providers/workers/useWorkers';
import LargeFileWorker from '@studio/workers/LargeFileWorker?worker';
import { useCallback } from 'react';

export interface DownloadFileAsArrayBufferArgs {
  workspace: string;
  datasetName: string;
  path: string;
}

/**
 * Spawns a LargeFileWorker to fetch a single file as an ArrayBuffer. Rejects with
 * the worker's error message on any failure. The worker self-terminates via
 * WorkersProvider when it signals `done` or errors.
 */
export function useFetchFileAsArrayBuffer() {
  const bearerToken = useOidcBearerToken();
  const { createWorker } = useWorkers();

  return useCallback(
    ({ workspace, datasetName, path }: DownloadFileAsArrayBufferArgs): Promise<ArrayBuffer> =>
      new Promise((resolve, reject) => {
        const worker = new LargeFileWorker();
        createWorker(worker, {
          onMessage: (e) => {
            const { done, arrayBuffer, error } = e.data;
            if (!done) return;
            if (arrayBuffer && !error) resolve(arrayBuffer);
            else reject(new Error(error || 'Could not download the file.'));
          },
          onError: (e) => reject(new Error(e.message || 'Could not download the file.')),
        });
        worker.postMessage({
          action: 'downloadAsFile',
          workspace,
          dataset: datasetName,
          path,
          accessToken: bearerToken,
        });
      }),
    [createWorker, bearerToken]
  );
}

/** {@link useFetchFileAsArrayBuffer}, resolving to `null` on any failure instead of rejecting. */
export function useDownloadFileAsArrayBuffer() {
  const fetchFile = useFetchFileAsArrayBuffer();

  return useCallback(
    (args: DownloadFileAsArrayBufferArgs): Promise<ArrayBuffer | null> =>
      fetchFile(args).catch(() => null),
    [fetchFile]
  );
}
