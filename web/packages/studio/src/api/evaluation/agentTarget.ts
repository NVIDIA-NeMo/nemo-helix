// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';

/** The registered agent a generic endpoint calls, read from the platform route it targets. Its
 *  `name` is only a display label, so the URL is what identifies the agent. */
const AGENT_ENDPOINT_PATH = /\/apis\/agents\/v2\/workspaces\/([^/]+)\/agents\/([^/]+)\/-\//;

const agentFromEndpointUrl = (url: unknown): string | null => {
  if (typeof url !== 'string') return null;
  const match = AGENT_ENDPOINT_PATH.exec(url);
  return match ? `${decodeURIComponent(match[1])}/${decodeURIComponent(match[2])}` : null;
};

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
  } else if (target.format === 'generic') {
    name = agentFromEndpointUrl(target.url);
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
