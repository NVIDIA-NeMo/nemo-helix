// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProgressBar } from '@nvidia/foundations-react-core';
import { CreateWorkerOptions, WorkersContextValue } from '@studio/providers/workers/types';
import { WorkersContext } from '@studio/providers/workers/WorkersContext';
import { FC, PropsWithChildren, useState } from 'react';

export const WorkersProvider: FC<PropsWithChildren> = ({ children }) => {
  const [workers, setWorkers] = useState<Set<Worker>>(new Set());

  const contextValue: WorkersContextValue = {
    workers,
    setWorkers,
    createWorker: (worker: Worker, options?: CreateWorkerOptions) => {
      worker.onmessage = (e) => {
        options?.onMessage?.(e);
        const { done } = e.data;
        if (done) {
          const newWorkers = new Set(workers);
          newWorkers.delete(worker);
          setWorkers(newWorkers);
          worker.terminate();
        }
      };
      worker.onerror = (e) => {
        options?.onError?.(e);
        const newWorkers = new Set(workers);
        newWorkers.delete(worker);
        setWorkers(newWorkers);
        worker.terminate();
      };
      const newWorkers = new Set(workers);
      newWorkers.add(worker);
      setWorkers(newWorkers);
    },
    terminateWorker: (worker: Worker) => {
      worker.terminate();
      setWorkers((current) => {
        const newWorkers = new Set(current);
        newWorkers.delete(worker);
        return newWorkers;
      });
    },
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
