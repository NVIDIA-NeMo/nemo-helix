// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_NAME_MAX_LENGTH, toValidEntityName } from '@nemo/common/src/utils/entityName';
import type { IntentId } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfig';

const STRIP_TRAILING_DASH = /-+$/;

const pad = (value: number): string => String(value).padStart(2, '0');

/** `MMDD-HHmmss` in local time. */
const timestamp = (now: Date): string =>
  `${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;

/** The name a study gets when the user has not typed one. */
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
