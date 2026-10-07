// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  useDownloadFileAsArrayBuffer,
  useFetchFileAsArrayBuffer,
} from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { useWorkers } from '@studio/providers/workers/useWorkers';
import { renderHook } from '@testing-library/react';

vi.mock('@studio/providers/workers/useWorkers');
vi.mock('react-oidc-context', () => ({
  useAuth: () => ({ user: { access_token: 'test-token' } }),
}));

// Captures the constructed worker so tests can assert the postMessage payload.
const postMessage = vi.fn();
const workerConstructor = vi.fn(() => ({ postMessage }));
vi.mock('@studio/workers/LargeFileWorker?worker', () => ({
  default: function LargeFileWorkerMock() {
    return workerConstructor();
  },
}));

const mockUseWorkers = vi.mocked(useWorkers);

// Captures the callbacks passed to createWorker so tests can synthesize
// `message` / `error` events without spinning up a real worker.
type Handlers = {
  onMessage: (e: { data: { done: boolean; arrayBuffer?: ArrayBuffer; error?: string } }) => void;
  onError?: (e: unknown) => void;
};
let handlers: Handlers | undefined;

describe('useDownloadFileAsArrayBuffer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    handlers = undefined;

    mockUseWorkers.mockReturnValue({
      createWorker: ((_worker: unknown, opts: Handlers) => {
        handlers = opts;
      }) as unknown as ReturnType<typeof useWorkers>['createWorker'],
    } as unknown as ReturnType<typeof useWorkers>);
  });

  const download = () =>
    renderHook(() => useDownloadFileAsArrayBuffer()).result.current({
      workspace: 'ws',
      datasetName: 'ds',
      path: 'data/a.txt',
    });

  it('posts the downloadAsFile message with workspace/dataset/path/accessToken', async () => {
    const promise = download();
    // Resolve so the test does not hang.
    handlers?.onMessage({ data: { done: true, arrayBuffer: new ArrayBuffer(1) } });
    await promise;

    expect(workerConstructor).toHaveBeenCalledTimes(1);
    expect(postMessage).toHaveBeenCalledWith({
      action: 'downloadAsFile',
      workspace: 'ws',
      dataset: 'ds',
      path: 'data/a.txt',
      accessToken: 'test-token',
    });
  });

  it('resolves to the arrayBuffer when the worker reports done', async () => {
    const buffer = new ArrayBuffer(4);

    const promise = download();
    handlers?.onMessage({ data: { done: true, arrayBuffer: buffer } });

    await expect(promise).resolves.toBe(buffer);
  });

  it('ignores progress messages (done: false) and only resolves on done', async () => {
    const promise = download();

    // Progress tick — must not resolve the promise.
    handlers?.onMessage({ data: { done: false } });
    const raceWinner = await Promise.race([
      promise,
      new Promise((res) => setTimeout(() => res('pending'), 10)),
    ]);
    expect(raceWinner).toBe('pending');

    // Final tick — now it resolves.
    handlers?.onMessage({ data: { done: true, arrayBuffer: new ArrayBuffer(2) } });
    await expect(promise).resolves.toBeInstanceOf(ArrayBuffer);
  });

  it('resolves to null when the worker reports an error payload', async () => {
    const promise = download();
    handlers?.onMessage({
      data: { done: true, arrayBuffer: new ArrayBuffer(2), error: 'boom' },
    });

    await expect(promise).resolves.toBeNull();
  });

  it('resolves to null when the worker reports done without a payload', async () => {
    const promise = download();
    handlers?.onMessage({ data: { done: true } });

    await expect(promise).resolves.toBeNull();
  });

  it('resolves to null when the worker fires onError', async () => {
    const promise = download();
    handlers?.onError?.(new Error('transport failure'));

    await expect(promise).resolves.toBeNull();
  });
});

describe('useFetchFileAsArrayBuffer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    handlers = undefined;

    mockUseWorkers.mockReturnValue({
      createWorker: ((_worker: unknown, opts: Handlers) => {
        handlers = opts;
      }) as unknown as ReturnType<typeof useWorkers>['createWorker'],
    } as unknown as ReturnType<typeof useWorkers>);
  });

  const fetchFile = () =>
    renderHook(() => useFetchFileAsArrayBuffer()).result.current({
      workspace: 'ws',
      datasetName: 'ds',
      path: 'data/a.txt',
    });

  it('resolves to the arrayBuffer when the worker reports done', async () => {
    const buffer = new ArrayBuffer(4);

    const promise = fetchFile();
    handlers?.onMessage({ data: { done: true, arrayBuffer: buffer } });

    await expect(promise).resolves.toBe(buffer);
  });

  it('rejects with the error the worker reports', async () => {
    const promise = fetchFile();
    handlers?.onMessage({ data: { done: true, error: 'Unable to find base file.' } });

    await expect(promise).rejects.toThrow('Unable to find base file.');
  });

  it('rejects when the worker fires onError', async () => {
    const promise = fetchFile();
    handlers?.onError?.(new ErrorEvent('error', { message: 'transport failure' }));

    await expect(promise).rejects.toThrow('transport failure');
  });
});
