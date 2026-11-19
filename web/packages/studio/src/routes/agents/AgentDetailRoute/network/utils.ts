// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AgentNetworkPolicy,
  AgentSandbox,
  NetworkRequest,
  NetworkRule,
} from '@studio/routes/agents/AgentDetailRoute/network/types';
import { stringify } from 'yaml';

const STATUS_ORDER: Record<string, number> = { pending: 0, approved: 1, rejected: 2 };

/** Requests waiting for a decision first, then newest activity first. */
export const sortRequests = (requests: NetworkRequest[]): NetworkRequest[] =>
  [...requests].sort(
    (a, b) =>
      (STATUS_ORDER[a.status] ?? 3) - (STATUS_ORDER[b.status] ?? 3) ||
      (b.last_seen ?? '').localeCompare(a.last_seen ?? '')
  );

const hasPending = (sandbox: AgentSandbox) =>
  sandbox.requests.some(({ status }) => status === 'pending');

const FINISHED_PHASES = new Set(['completed', 'error', 'deleting']);

/** Whether the sandbox's workload is over, so a change to its policy no longer reaches anything. */
export const isFinished = (sandbox: AgentSandbox) => FINISHED_PHASES.has(sandbox.phase);

/** Sandboxes with something to decide first, then those still running, then finished ones. */
export const sortSandboxes = (sandboxes: AgentSandbox[]): AgentSandbox[] =>
  [...sandboxes].sort(
    (a, b) =>
      Number(hasPending(b)) - Number(hasPending(a)) ||
      Number(isFinished(a)) - Number(isFinished(b)) ||
      a.name.localeCompare(b.name)
  );

export const pendingCount = (policy: AgentNetworkPolicy | undefined): number =>
  (policy?.sandboxes ?? []).reduce(
    (count, sandbox) =>
      count + sandbox.requests.filter(({ status }) => status === 'pending').length,
    0
  );

/** The destinations of a request or rule, e.g. "example.com:443". */
export const requestTarget = (request: NetworkRequest): string =>
  request.endpoints.join(', ') || request.rule_name;

export const ruleEndpoints = (rule: NetworkRule): string[] =>
  (rule.endpoints ?? []).map(({ host, port }) => `${host ?? '?'}:${port ?? '?'}`);

export const ruleBinaries = (rule: NetworkRule): string[] =>
  (rule.binaries ?? []).flatMap(({ path }) => (path ? [path] : []));

/** The last path segment, which is how a person names an executable. */
export const binaryName = (path: string): string => path.split('/').pop() || path;

/** One rule as it would be written in agent.yaml. */
export const agentYamlFragment = (name: string, rule: NetworkRule): string =>
  stringify({ helix_registration: { network_policies: { [name]: rule } } });
