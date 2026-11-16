// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DatetimeFilter, Taskset } from '@nemo/sdk/generated/evals/schema';

/** A task reference split into its parts: `[workspace/]name[#revision]`. */
export interface TaskRefParts {
  workspace: string;
  name: string;
  /** Pinned revision digest or tag; absent when the ref follows the latest revision. */
  version?: string;
}

/** Task list column filters, keyed by column id. */
export interface TaskListFilters {
  /** Multi-select state: the selected runner kinds. */
  runner?: Record<string, true>;
  tags?: string;
  updated_at?: DatetimeFilter;
}

/** A taskset that has the task as a member, with the revision it pins. */
export interface TaskUsageRow extends Taskset {
  /** The member ref as the taskset stores it, pin included. */
  memberRef: string;
  /** Content digest of the pinned task revision; absent when the member ref is unpinned. */
  pinnedDigest?: string;
  /** Ordinal of the pinned revision, when it is one of the task's listed revisions. */
  pinnedRevision?: number;
  /** Ordinal of the task's latest revision, once its revisions load. */
  latestRevision?: number;
  /** The task has a revision newer than the pin; undefined when its revisions couldn't be
   *  loaded to compare against. */
  hasNewerRevision?: boolean;
}
