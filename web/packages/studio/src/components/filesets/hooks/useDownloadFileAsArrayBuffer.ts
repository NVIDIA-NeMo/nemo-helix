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
  signal?: AbortSignal;
}

const abortError = (signal: AbortSignal): unknown =>
  signal.reason ?? new DOMException('The download was aborted.', 'AbortError');

/**
 * Spawns a LargeFileWorker to fetch a single file as an ArrayBuffer. Rejects with
 * the worker's error message on any failure. The worker self-terminates via
 * WorkersProvider when it signals `done` or errors; aborting `signal` terminates it
 * and rejects with the abort reason.
 */
export function useFetchFileAsArrayBuffer() {
  const bearerToken = useOidcBearerToken();
  const { createWorker, terminateWorker } = useWorkers();

  return useCallback(
    ({
      workspace,
      datasetName,
      path,
      signal,
    }: DownloadFileAsArrayBufferArgs): Promise<ArrayBuffer> =>
      new Promise((resolve, reject) => {
        if (signal?.aborted) {
          reject(abortError(signal));
          return;
        }
        const worker = new LargeFileWorker();
        const onAbort = () => {
          terminateWorker(worker);
          if (signal) reject(abortError(signal));
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        const settle = () => signal?.removeEventListener('abort', onAbort);

        createWorker(worker, {
          onMessage: (e) => {
            const { done, arrayBuffer, error } = e.data;
            if (!done) return;
            settle();
            if (arrayBuffer && !error) resolve(arrayBuffer);
            else reject(new Error(error || 'Could not download the file.'));
          },
          onError: (e) => {
            settle();
            reject(new Error(e.message || 'Could not download the file.'));
          },
        });
        worker.postMessage({
          action: 'downloadAsFile',
          workspace,
          dataset: datasetName,
          path,
          accessToken: bearerToken,
        });
      }),
    [createWorker, terminateWorker, bearerToken]
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
