// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvalsListTaskRevisions } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import { useEvalsListTasksets } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import { taskRef } from '@studio/routes/evaluation/taskHelpers';
import { toTaskUsageRows } from '@studio/routes/evaluation/tasks/TaskDetailRoute/taskUsageRows';
import { keepPreviousData } from '@tanstack/react-query';
import { useMemo } from 'react';

/** Revisions fetched to name pinned digests; an older pin shows its short digest instead. */
const REVISION_LOOKUP_SIZE = 100;

/** One page of the tasksets in the task's workspace that have it as a member, each with the
 *  revision it pins and whether the task has a newer one. */
export function useTaskUsage(task: Task, page: { page?: number; page_size: number }) {
  const tasksets = useEvalsListTasksets(
    task.workspace,
    { ...page, filter: { tasks: taskRef(task) } },
    { query: { placeholderData: keepPreviousData } }
  );
  const { data: revisions, isError: revisionsFailed } = useEvalsListTaskRevisions(
    task.workspace,
    task.name,
    { page_size: REVISION_LOOKUP_SIZE }
  );

  const rows = useMemo(
    () =>
      toTaskUsageRows({
        task,
        tasksets: tasksets.data?.data ?? [],
        revisions: revisions?.data,
        revisionsFailed,
      }),
    [task, tasksets.data, revisions, revisionsFailed]
  );

  return {
    rows,
    totalCount: tasksets.data?.pagination?.total_results,
    isFetching: tasksets.isFetching,
    error: tasksets.error,
  };
}
