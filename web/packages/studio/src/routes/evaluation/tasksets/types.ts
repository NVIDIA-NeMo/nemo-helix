// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';

/** A task reference split into its parts: `[workspace/]name[#revision]`. */
export interface TaskRefParts {
  workspace: string;
  name: string;
  /** Pinned revision digest or tag; absent when the ref follows the latest revision. */
  version?: string;
}

/** One taskset member: its stored ref, and the task record when it resolved. */
export interface TasksetTaskRow extends TaskRefParts {
  id: string;
  ref: string;
  task?: Task;
}

/** `workspace/fileset#path`, split. */
export interface FilesRefParts {
  workspace: string;
  fileset: string;
  path?: string;
}
