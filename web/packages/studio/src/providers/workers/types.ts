// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

export interface CreateWorkerOptions {
  onMessage?: (e: MessageEvent) => void;
  onError?: (e: ErrorEvent) => void;
}

export interface WorkersContextValue {
  workers: Set<Worker>;
  setWorkers: Dispatch<SetStateAction<Set<Worker>>>;
  createWorker: (worker: Worker, options?: CreateWorkerOptions) => void;
  terminateWorker: (worker: Worker) => void;
}
