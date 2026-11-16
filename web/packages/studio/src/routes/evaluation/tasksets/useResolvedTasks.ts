// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import {
  type evalsListTasks,
  getEvalsListTasksQueryOptions,
} from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import type { Task, TaskFilter } from '@nemo/sdk/generated/evals/schema';
import { taskRef } from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { type UseQueryResult, useQueries } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

/** Names per list call: keeps each `$in` filter well under URL length limits. */
const BATCH_SIZE = 50;

type BatchResult = UseQueryResult<Awaited<ReturnType<typeof evalsListTasks>>>;

/**
 * Attaches task records to `rows`, in batched list calls per workspace so a taskset with
 * thousands of refs never overflows the URL. A pinned ref resolves to the task's current
 * record, not the pinned revision: fetching each pin's revision would cost a request per member.
 * A failed batch marks only its own rows `failed`; `error` is set once every batch has failed.
 */
export function useResolvedTasks(rows: TasksetTaskRow[]) {
  const batches = useMemo(() => {
    const grouped = new Map<string, Set<string>>();
    rows.forEach(({ workspace, name }) => {
      if (!grouped.has(workspace)) grouped.set(workspace, new Set());
      grouped.get(workspace)?.add(name);
    });
    return [...grouped].flatMap(([workspace, names]) => {
      const unique = [...names];
      return Array.from({ length: Math.ceil(unique.length / BATCH_SIZE) }, (_, i) => ({
        workspace,
        names: unique.slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE),
      }));
    });
  }, [rows]);

  const combine = useCallback(
    (results: BatchResult[]) => {
      const byRef = new Map<string, Task>();
      const failed = new Set<string>();
      results.forEach(({ data, error }, i) => {
        const { workspace, names } = batches[i];
        if (error) names.forEach((name) => failed.add(taskRef({ workspace, name })));
        data?.data.forEach((task) => byRef.set(taskRef(task), task));
      });
      const errors = results.flatMap(({ error }) => (error ? [error] : []));
      return {
        byRef,
        failed,
        isFetching: results.some(({ isFetching }) => isFetching),
        error: errors.length > 0 && errors.length === results.length ? errors[0] : null,
      };
    },
    [batches]
  );

  const { byRef, failed, isFetching, error } = useQueries({
    queries: batches.map(({ workspace, names }) =>
      getEvalsListTasksQueryOptions(workspace, {
        page_size: names.length,
        filter: withOperators<TaskFilter>({ name: { $in: names } }),
      })
    ),
    combine,
  });

  const resolved = useMemo(
    () =>
      rows.map((row) => ({
        ...row,
        task: byRef.get(taskRef(row)),
        failed: failed.has(taskRef(row)),
      })),
    [rows, byRef, failed]
  );

  return { rows: resolved, isFetching, error };
}
