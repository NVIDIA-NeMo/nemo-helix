// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { getEvalsListTasksQueryOptions } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import type { Task, TaskFilter } from '@nemo/sdk/generated/evals/schema';
import { taskRef } from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

/**
 * Attaches task records to `rows`, one list call per workspace. Pass only the rows on screen:
 * a taskset can hold thousands of refs, and one `$in` filter over all of them overflows the URL.
 * A pinned ref resolves to the task's current record, not the pinned revision.
 */
export function useResolvedTasks(rows: TasksetTaskRow[]) {
  const namesByWorkspace = useMemo(() => {
    const grouped = new Map<string, string[]>();
    rows.forEach(({ workspace, name }) =>
      grouped.set(workspace, [...(grouped.get(workspace) ?? []), name])
    );
    return [...grouped];
  }, [rows]);

  const results = useQueries({
    queries: namesByWorkspace.map(([workspace, names]) =>
      getEvalsListTasksQueryOptions(workspace, {
        page_size: names.length,
        filter: withOperators<TaskFilter>({ name: { $in: names } }),
      })
    ),
  });

  const byRef = new Map<string, Task>(
    results.flatMap((result) => result.data?.data ?? []).map((task) => [taskRef(task), task])
  );

  return {
    rows: rows.map((row) => ({ ...row, task: byRef.get(taskRef(row)) })),
    isFetching: results.some((result) => result.isFetching),
    error: results.find((result) => result.error)?.error ?? null,
  };
}
