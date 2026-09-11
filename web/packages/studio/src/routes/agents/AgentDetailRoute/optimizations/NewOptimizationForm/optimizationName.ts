// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_NAME_MAX_LENGTH, toValidEntityName } from '@nemo/common/src/utils/entityName';
import type { IntentId } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfig';

const STRIP_TRAILING_DASH = /-+$/;

const pad = (value: number): string => String(value).padStart(2, '0');

/** `MMDD-HHmmss` in local time — short enough to read in a table, precise enough that two runs
 *  started back to back do not collide. */
const timestamp = (now: Date): string =>
  `${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

/**
 * The name a study gets when the user has not typed one.
 *
 * `<agent>-<intent>-<timestamp>` — the intent is in the name because it is the one thing that
 * distinguishes two studies on the same agent, and it is what a user scanning the studies table
 * is actually looking for. The agent segment is truncated, not the suffix, so the part that
 * disambiguates always survives `ENTITY_NAME_MAX_LENGTH`.
 */
export const buildOptimizationName = (
  agentName: string,
  intent: IntentId,
  now: Date = new Date()
): string => {
  const suffix = `${intent}-${timestamp(now)}`;
  const base = agentName
    .slice(0, ENTITY_NAME_MAX_LENGTH - suffix.length - 1)
    .replace(STRIP_TRAILING_DASH, '');

  return toValidEntityName(`${base}-${suffix}`, suffix);
};
