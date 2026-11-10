// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEvalsListTaskRevisions } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import { useEvalsListTasksets } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import type { TaskUsageRow } from '@studio/routes/evaluation/tasks/types';
import { parseTaskRef, taskRef } from '@studio/routes/evaluation/tasksets/helpers';
import { keepPreviousData } from '@tanstack/react-query';
import { useMemo } from 'react';

/** Revisions fetched to name pinned digests; an older pin shows its short digest instead. */
const REVISION_LOOKUP_SIZE = 100;

/** One page of the tasksets in the task's workspace that have it as a member, each with the
 *  revision it pins and whether that pin is behind the task's latest content. */
export function useTaskUsage(task: Task, page: { page?: number; page_size: number }) {
  const ref = taskRef(task);
  const tasksets = useEvalsListTasksets(
    task.workspace,
    { ...page, filter: { tasks: ref } },
    { query: { placeholderData: keepPreviousData } }
  );
  const { data: revisions } = useEvalsListTaskRevisions(task.workspace, task.name, {
    page_size: REVISION_LOOKUP_SIZE,
  });

  const rows = useMemo<TaskUsageRow[]>(() => {
    // Newest first, so a digest published twice (a revert) names its newest revision.
    const latestDigest = revisions?.data[0]?.content_hash;
    const ordinals = new Map<string, number>();
    revisions?.data.forEach(({ content_hash, revision }) => {
      if (!ordinals.has(content_hash)) ordinals.set(content_hash, revision);
    });
    return (tasksets.data?.data ?? []).map((taskset) => {
      const pinnedDigest = (taskset.tasks ?? [])
        .map((member) => parseTaskRef(member, taskset.workspace))
        .find((parts) => taskRef(parts) === ref)?.version;
      return {
        ...taskset,
        pinnedDigest,
        pinnedRevision: pinnedDigest ? ordinals.get(pinnedDigest) : undefined,
        behindLatest: Boolean(pinnedDigest && latestDigest && pinnedDigest !== latestDigest),
      };
    });
  }, [tasksets.data, revisions, ref]);

  return {
    ref,
    rows,
    totalCount: tasksets.data?.pagination?.total_results,
    isFetching: tasksets.isFetching,
    error: tasksets.error,
  };
}
