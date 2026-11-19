// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import type { AgentNetworkPolicy } from '@studio/routes/agents/AgentDetailRoute/network/types';

const networkPolicyPath = (workspace: string, agent: string) =>
  `/apis/agents/v2/workspaces/${encodeURIComponent(workspace)}/agents/${encodeURIComponent(agent)}/network-policy`;

const requestPath = (workspace: string, agent: string, sandbox: string, requestId: string) =>
  `${networkPolicyPath(workspace, agent)}/sandboxes/${encodeURIComponent(sandbox)}/requests/${encodeURIComponent(requestId)}`;

const errorDetail = async (response: Response): Promise<string> => {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
  } catch {
    // Not JSON; the status is all there is.
  }
  return `Request failed: ${response.status}`;
};

const requestPolicy = async (path: string, init?: RequestInit): Promise<AgentNetworkPolicy> => {
  const response = await platformFetch(path, init);
  if (!response.ok) {
    throw new Error(await errorDetail(response));
  }
  return (await response.json()) as AgentNetworkPolicy;
};

const post = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const getAgentNetworkPolicy = (workspace: string, agent: string) =>
  requestPolicy(networkPolicyPath(workspace, agent));

export const approveNetworkRequest = (
  workspace: string,
  agent: string,
  sandbox: string,
  requestId: string,
  remember: boolean
) =>
  requestPolicy(`${requestPath(workspace, agent, sandbox, requestId)}/approve`, post({ remember }));

export const rejectNetworkRequest = (
  workspace: string,
  agent: string,
  sandbox: string,
  requestId: string,
  options: { reason: string; forget: boolean }
) => requestPolicy(`${requestPath(workspace, agent, sandbox, requestId)}/reject`, post(options));

export const forgetNetworkRule = (workspace: string, agent: string, rule: string) =>
  requestPolicy(`${networkPolicyPath(workspace, agent)}/rules/${encodeURIComponent(rule)}`, {
    method: 'DELETE',
  });
