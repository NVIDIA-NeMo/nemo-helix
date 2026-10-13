// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WorkersContextValue } from '@studio/providers/workers/types';
import { useWorkers } from '@studio/providers/workers/useWorkers';
import { WorkersProvider } from '@studio/providers/workers/WorkersProvider';
import { act, render, screen } from '@testing-library/react';

const fakeWorker = () =>
  ({ terminate: vi.fn(), onmessage: null, onerror: null }) as unknown as Worker;

const message = (data: unknown) => new MessageEvent('message', { data });

const mountProvider = () => {
  let context: WorkersContextValue | undefined;
  const Capture = () => {
    context = useWorkers();
    return null;
  };
  render(
    <WorkersProvider>
      <Capture />
    </WorkersProvider>
  );
  return () => context as WorkersContextValue;
};

const progressBar = () => screen.queryByLabelText('Worker progress');

describe('WorkersProvider', () => {
  it('shows progress while a worker runs and hides it when the worker is done', () => {
    const workers = mountProvider();
    const worker = fakeWorker();

    act(() => workers().createWorker(worker));
    expect(progressBar()).toBeInTheDocument();

    act(() => worker.onmessage?.(message({ done: true })));
    expect(progressBar()).not.toBeInTheDocument();
    expect(worker.terminate).toHaveBeenCalled();
  });

  it('removes a worker that errors', () => {
    const workers = mountProvider();
    const worker = fakeWorker();

    act(() => workers().createWorker(worker));
    act(() => worker.onerror?.(new ErrorEvent('error')));

    expect(progressBar()).not.toBeInTheDocument();
    expect(worker.terminate).toHaveBeenCalled();
  });

  it('does not restore a terminated worker when another worker finishes', () => {
    const workers = mountProvider();
    const aborted = fakeWorker();
    const finishing = fakeWorker();

    act(() => workers().createWorker(aborted));
    act(() => workers().createWorker(finishing));
    act(() => workers().terminateWorker(aborted));
    act(() => finishing.onmessage?.(message({ done: true })));

    expect(progressBar()).not.toBeInTheDocument();
  });
});
