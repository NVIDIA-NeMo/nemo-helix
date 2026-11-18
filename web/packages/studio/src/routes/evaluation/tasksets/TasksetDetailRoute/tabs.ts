// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Taskset detail tabs, each its own path segment; the first is the bare detail route. */
export const TASKSET_DETAIL_TABS = [
  { value: 'tasks', label: 'Tasks' },
  { value: 'revisions', label: 'Revisions' },
] as const;

export type TasksetDetailTab = (typeof TASKSET_DETAIL_TABS)[number]['value'];

export const isTasksetDetailTab = (value: string | undefined): value is TasksetDetailTab =>
  TASKSET_DETAIL_TABS.some((tab) => tab.value === value);
