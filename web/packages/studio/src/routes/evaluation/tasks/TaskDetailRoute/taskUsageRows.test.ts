// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Revision, Taskset } from '@nemo/sdk/generated/evals/schema';
import { toTaskUsageRows } from '@studio/routes/evaluation/tasks/TaskDetailRoute/taskUsageRows';

const task = { workspace: 'ws', name: 'sort' };
const revision = (n: number, content_hash: string) =>
  ({ revision: n, content_hash, created_at: '2026-10-01T09:00:00Z' }) as Revision;
const taskset = (name: string, tasks: string[]) =>
  ({ id: name, name, workspace: 'ws', tasks }) as Taskset;
// Newest first; r3 reverts to r1's content.
const revisions = [revision(3, 'aaa'), revision(2, 'bbb'), revision(1, 'aaa')];

describe('toTaskUsageRows', () => {
  it('matches the member across ref spellings and names its pinned revision', () => {
    const [row] = toTaskUsageRows({
      task,
      tasksets: [taskset('suite', ['other', 'sort#bbb'])],
      revisions,
      revisionsFailed: false,
    });
    expect(row).toMatchObject({
      memberRef: 'sort#bbb',
      pinnedDigest: 'bbb',
      pinnedRevision: 2,
      latestRevision: 3,
      hasNewerRevision: true,
    });
  });

  it('names a reverted digest by its newest revision, which is latest', () => {
    const [row] = toTaskUsageRows({
      task,
      tasksets: [taskset('suite', ['ws/sort#aaa'])],
      revisions,
      revisionsFailed: false,
    });
    expect(row).toMatchObject({ pinnedRevision: 3, hasNewerRevision: false });
  });

  it('never flags an unpinned member', () => {
    const [row] = toTaskUsageRows({
      task,
      tasksets: [taskset('suite', ['ws/sort'])],
      revisionsFailed: true,
    });
    expect(row).toMatchObject({ pinnedDigest: undefined, hasNewerRevision: false });
  });

  it.each([
    [false, false],
    [true, undefined],
  ])('without revisions (failed: %s) a pin compares as %s', (revisionsFailed, expected) => {
    const [row] = toTaskUsageRows({
      task,
      tasksets: [taskset('suite', ['ws/sort#bbb'])],
      revisionsFailed,
    });
    expect(row.hasNewerRevision).toBe(expected);
    expect(row.pinnedRevision).toBeUndefined();
  });
});
