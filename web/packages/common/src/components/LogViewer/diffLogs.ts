// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

export interface LogsDiff {
  /** How many entries fell off the head of the previous list. */
  dropped: number;
  /** The entries that follow the ones the two lists share. */
  added: HelixJobLog[];
}

// By content, not identity: polls refetch the last page as fresh objects, and a trimmed
// head shifts indexes so React Query's structural sharing can't reuse the old ones.
const isSameEntry = (a: HelixJobLog, b: HelixJobLog): boolean =>
  a.timestamp === b.timestamp && a.message === b.message;

/** Describes `next` as `prev` with entries dropped from its head and appended to its tail,
 *  or returns null when they don't overlap that way and the content must be replaced. */
export function diffLogs(
  prev: readonly HelixJobLog[],
  next: readonly HelixJobLog[]
): LogsDiff | null {
  if (prev.length === 0) return { dropped: 0, added: next.slice() };

  // Lines repeat in logs, so the first match of next[0] isn't enough: the rest of prev
  // has to line up with next too. Trying the smallest drop first keeps the most content.
  for (let dropped = 0; dropped < prev.length; dropped++) {
    const shared = prev.length - dropped;
    if (shared > next.length) continue;
    // Checking both ends of the overlap first rules out most candidates cheaply.
    if (!isSameEntry(prev[dropped], next[0])) continue;
    if (!isSameEntry(prev[prev.length - 1], next[shared - 1])) continue;

    let matches = true;
    for (let i = 1; i < shared - 1; i++) {
      if (!isSameEntry(prev[dropped + i], next[i])) {
        matches = false;
        break;
      }
    }
    if (matches) return { dropped, added: next.slice(shared) };
  }

  return null;
}
