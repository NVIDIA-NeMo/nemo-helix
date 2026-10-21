// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Puts `owned` ahead of `all`, dropping the copies of owned items that `all` also contains
 * (matched by `id`). Each list keeps its own order, so newest-first inputs stay newest-first.
 * An owned item that `all` missed (for example it fell off the end of a page) is still included.
 */
export const ownedFirst = <T extends { id: string }>(
  owned: readonly T[],
  all: readonly T[]
): T[] => {
  const ownedIds = new Set(owned.map((item) => item.id));
  return [...owned, ...all.filter((item) => !ownedIds.has(item.id))];
};
