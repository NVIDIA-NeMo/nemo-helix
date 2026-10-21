// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Moves the signed-in user's own items (`created_by === ownerId`) ahead of everyone else's.
 * Stable: each group keeps the order it arrived in, so a newest-first list stays newest-first.
 * Returns the list untouched when there is no signed-in user.
 */
export const ownedFirst = <T extends { created_by?: string | null }>(
  items: readonly T[],
  ownerId: string | undefined
): T[] =>
  ownerId
    ? [
        ...items.filter((item) => item.created_by === ownerId),
        ...items.filter((item) => item.created_by !== ownerId),
      ]
    : [...items];
