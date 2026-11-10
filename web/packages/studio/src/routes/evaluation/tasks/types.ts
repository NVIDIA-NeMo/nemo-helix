// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DatetimeFilter, Taskset } from '@nemo/sdk/generated/evals/schema';

/** Task list column filters, keyed by column id. */
export interface TaskListFilters {
  /** Multi-select state: the selected runner kinds. */
  runner?: Record<string, true>;
  tags?: string;
  updated_at?: DatetimeFilter;
}

/** A taskset that has the task as a member, with the revision it pins. */
export interface TaskUsageRow extends Taskset {
  /** Content digest of the pinned task revision; absent when the member ref is unpinned. */
  pinnedDigest?: string;
  /** Ordinal of the pinned revision, when it is one of the task's listed revisions. */
  pinnedRevision?: number;
  /** The pin names content other than the task's latest revision. */
  behindLatest: boolean;
}
