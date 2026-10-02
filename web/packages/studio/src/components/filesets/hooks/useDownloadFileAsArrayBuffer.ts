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
 * Spawns a LargeFileWorker to fetch a single file as an ArrayBuffer. Resolves
 * to `null` on any failure (transport error, worker error, missing payload).
 * The worker self-terminates via WorkersProvider when it signals `done` or errors.
 */
export function useDownloadFileAsArrayBuffer() {
  const bearerToken = useOidcBearerToken();
  const { createWorker } = useWorkers();

  return useCallback(
    ({
      workspace,
      datasetName,
      path,
    }: DownloadFileAsArrayBufferArgs): Promise<ArrayBuffer | null> =>
      new Promise((resolve) => {
        const worker = new LargeFileWorker();
        createWorker(worker, {
          onMessage: (e) => {
            const { done, arrayBuffer, error } = e.data;
            if (!done) return;
            resolve(arrayBuffer && !error ? arrayBuffer : null);
          },
          onError: () => resolve(null),
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
