// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Task detail tabs, each its own path segment; the first is the bare detail route. */
export const TASK_DETAIL_TABS = [
  { value: 'overview', label: 'Overview' },
  { value: 'revisions', label: 'Revisions' },
  { value: 'used-by', label: 'Used By' },
] as const;

export type TaskDetailTab = (typeof TASK_DETAIL_TABS)[number]['value'];

export const isTaskDetailTab = (value: string | undefined): value is TaskDetailTab =>
  TASK_DETAIL_TABS.some((tab) => tab.value === value);
