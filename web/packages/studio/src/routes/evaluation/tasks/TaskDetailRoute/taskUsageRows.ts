// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Revision, Task, Taskset } from '@nemo/sdk/generated/evals/schema';
import { parseTaskRef, taskRef } from '@studio/routes/evaluation/taskHelpers';
import type { TaskUsageRow } from '@studio/routes/evaluation/tasks/types';

interface TaskUsageInput {
  task: Pick<Task, 'workspace' | 'name'>;
  tasksets: Taskset[];
  /** The task's revisions, newest first; undefined until (or unless) they load. */
  revisions?: Revision[];
  revisionsFailed: boolean;
}

/** Each taskset's member ref for the task, the revision it pins, and whether a newer one exists. */
export function toTaskUsageRows({
  task,
  tasksets,
  revisions = [],
  revisionsFailed,
}: TaskUsageInput): TaskUsageRow[] {
  const ref = taskRef(task);
  const latest = revisions[0];
  // Newest first, so a digest published twice (a revert) names its newest revision.
  const ordinals = new Map<string, number>();
  revisions.forEach(({ content_hash, revision }) => {
    if (!ordinals.has(content_hash)) ordinals.set(content_hash, revision);
  });
  // Unknown only once the lookup has failed; while it loads, nothing is flagged.
  const hasNewer = (pinnedDigest?: string): boolean | undefined => {
    if (!pinnedDigest) return false;
    if (latest) return pinnedDigest !== latest.content_hash;
    return revisionsFailed ? undefined : false;
  };
  return tasksets.map((taskset) => {
    const memberRef =
      taskset.tasks?.find((member) => taskRef(parseTaskRef(member, taskset.workspace)) === ref) ??
      ref;
    const pinnedDigest = parseTaskRef(memberRef, taskset.workspace).version;
    return {
      ...taskset,
      memberRef,
      pinnedDigest,
      pinnedRevision: pinnedDigest ? ordinals.get(pinnedDigest) : undefined,
      latestRevision: latest?.revision,
      hasNewerRevision: hasNewer(pinnedDigest),
    };
  });
}
