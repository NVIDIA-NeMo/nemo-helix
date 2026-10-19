// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProgressBar } from '@nvidia/foundations-react-core';
import { CreateWorkerOptions, WorkersContextValue } from '@studio/providers/workers/types';
import { WorkersContext } from '@studio/providers/workers/WorkersContext';
import { FC, PropsWithChildren, useState } from 'react';

export const WorkersProvider: FC<PropsWithChildren> = ({ children }) => {
  const [workers, setWorkers] = useState<Set<Worker>>(new Set());

  const terminateWorker = (worker: Worker) => {
    worker.terminate();
    setWorkers((current) => {
      const newWorkers = new Set(current);
      newWorkers.delete(worker);
      return newWorkers;
    });
  };

  const contextValue: WorkersContextValue = {
    workers,
    setWorkers,
    createWorker: (worker: Worker, options?: CreateWorkerOptions) => {
      worker.onmessage = (e) => {
        options?.onMessage?.(e);
        if (e.data.done) terminateWorker(worker);
      };
      worker.onerror = (e) => {
        options?.onError?.(e);
        terminateWorker(worker);
      };
      setWorkers((current) => new Set(current).add(worker));
    },
    terminateWorker,
  };

  return (
    <WorkersContext.Provider value={contextValue}>
      {workers.size !== 0 && (
        <ProgressBar
          kind="indeterminate"
          className="absolute top-0 left-0 right-0 z-[1000] rounded-none"
          aria-label="Worker progress"
        />
      )}
      {children}
    </WorkersContext.Provider>
  );
};
