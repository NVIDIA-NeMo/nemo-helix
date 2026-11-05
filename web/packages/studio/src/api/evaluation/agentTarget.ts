// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@studio/util/functions';

/** Resolve only targets that identify a registered agent, never a runner's display name.
 * Registered source AgentRefs serialize as strings, not objects with a `name` field. */
export const agentIdentityForTarget = (target: unknown, workspace?: string): string | null => {
  if (!isPlainObject(target)) return null;
  let name: unknown;
  if (target.kind === 'agent' && isPlainObject(target.agent)) {
    name = target.agent.name;
  } else if (
    (target.kind === 'fabric' || target.kind === 'harbor' || target.kind === 'gym') &&
    isPlainObject(target.source)
  ) {
    name = target.source.agent;
  }
  if (typeof name !== 'string' || name.trim().length === 0) return null;
  return name.includes('/') || !workspace ? name : `${workspace}/${name}`;
};

/** Keep foreign-workspace refs qualified so they cannot match a local agent of the same name. */
export const agentNameForTarget = (target: unknown, workspace?: string): string | null => {
  const identity = agentIdentityForTarget(target, workspace);
  const prefix = `${workspace}/`;
  return workspace && identity?.startsWith(prefix) ? identity.slice(prefix.length) : identity;
};
