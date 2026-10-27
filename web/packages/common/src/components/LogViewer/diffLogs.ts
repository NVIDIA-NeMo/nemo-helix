// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isSameLogEntry } from '@nemo/common/src/utils/logs';
import type { HelixJobLog } from '@nemo/sdk/generated/platform/schema';

export interface LogsDiff {
  /** Older entries loaded in front of the previous list. */
  prepended: HelixJobLog[];
  /** How many entries fell off the head of the previous list. */
  dropped: number;
  /** The entries that follow the ones the two lists share. */
  added: HelixJobLog[];
}

/** Whether `length` entries of `prev` from `prevStart` match those of `next` from `nextStart`. */
function alignsAt(
  prev: readonly HelixJobLog[],
  prevStart: number,
  next: readonly HelixJobLog[],
  nextStart: number,
  length: number
): boolean {
  // Checking both ends first rules out most candidates cheaply.
  const last = length - 1;
  if (!isSameLogEntry(prev[prevStart], next[nextStart])) {
    return false;
  }
  if (!isSameLogEntry(prev[prevStart + last], next[nextStart + last])) {
    return false;
  }
  for (let i = 1; i < last; i++) {
    if (!isSameLogEntry(prev[prevStart + i], next[nextStart + i])) {
      return false;
    }
  }
  return true;
}

/** Describes `next` as `prev` with entries dropped from or prepended to its head and appended
 *  to its tail, or returns null when they don't overlap that way and the content must be
 *  replaced. */
export function diffLogs(
  prev: readonly HelixJobLog[],
  next: readonly HelixJobLog[]
): LogsDiff | null {
  if (prev.length === 0) {
    return { prepended: [], dropped: 0, added: next.slice() };
  }

  // Lines repeat in logs, so the first match of next[0] isn't enough: the rest of prev
  // has to line up with next too. Trying the smallest drop first keeps the most content.
  for (let dropped = 0; dropped < prev.length; dropped++) {
    const shared = prev.length - dropped;
    if (shared <= next.length && alignsAt(prev, dropped, next, 0, shared)) {
      return { prepended: [], dropped, added: next.slice(shared) };
    }
  }

  // Older entries loaded in front: all of prev reappears further into next.
  for (let prepended = 1; prepended + prev.length <= next.length; prepended++) {
    if (alignsAt(prev, 0, next, prepended, prev.length)) {
      return {
        prepended: next.slice(0, prepended),
        dropped: 0,
        added: next.slice(prepended + prev.length),
      };
    }
  }

  return null;
}
