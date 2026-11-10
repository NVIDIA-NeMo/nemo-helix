// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { getEvalsListTasksQueryOptions } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import type { Task, TaskFilter } from '@nemo/sdk/generated/evals/schema';
import { taskRef } from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

/** Names per list call: keeps each `$in` filter well under URL length limits. */
const BATCH_SIZE = 100;

/**
 * Attaches task records to `rows`, in batched list calls per workspace so a taskset with
 * thousands of refs never overflows the URL. A pinned ref resolves to the task's current
 * record, not the pinned revision.
 */
export function useResolvedTasks(rows: TasksetTaskRow[]) {
  const batches = useMemo(() => {
    const grouped = new Map<string, string[]>();
    rows.forEach(({ workspace, name }) =>
      grouped.set(workspace, [...(grouped.get(workspace) ?? []), name])
    );
    return [...grouped].flatMap(([workspace, names]) => {
      const unique = [...new Set(names)];
      return Array.from({ length: Math.ceil(unique.length / BATCH_SIZE) }, (_, i) => ({
        workspace,
        names: unique.slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE),
      }));
    });
  }, [rows]);

  const results = useQueries({
    queries: batches.map(({ workspace, names }) =>
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
