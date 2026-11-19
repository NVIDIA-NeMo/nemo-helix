// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  approveNetworkRequest,
  forgetNetworkRule,
  getAgentNetworkPolicy,
  rejectNetworkRequest,
} from '@studio/routes/agents/AgentDetailRoute/network/api';
import type { AgentNetworkPolicy } from '@studio/routes/agents/AgentDetailRoute/network/types';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

/** A blocked sandbox files its request about 12 seconds after the first denial. */
const POLL_INTERVAL_MS = 5000;

const networkPolicyQueryKey = (workspace: string, agent: string) =>
  ['agents', workspace, agent, 'network-policy'] as const;

export const useAgentNetworkPolicy = (
  workspace: string,
  agent: string | undefined,
  { enabled = true }: { enabled?: boolean } = {}
) =>
  useQuery({
    queryKey: networkPolicyQueryKey(workspace, agent ?? ''),
    queryFn: () => getAgentNetworkPolicy(workspace, agent ?? ''),
    enabled: enabled && !!workspace && !!agent,
    refetchInterval: POLL_INTERVAL_MS,
  });

export type NetworkDecision =
  | { action: 'approve'; sandbox: string; requestId: string; remember: boolean }
  | { action: 'reject'; sandbox: string; requestId: string; reason: string; forget: boolean }
  | { action: 'forget'; rule: string };

const decide = (
  workspace: string,
  agent: string,
  decision: NetworkDecision
): Promise<AgentNetworkPolicy> => {
  switch (decision.action) {
    case 'approve':
      return approveNetworkRequest(
        workspace,
        agent,
        decision.sandbox,
        decision.requestId,
        decision.remember
      );
    case 'reject':
      return rejectNetworkRequest(workspace, agent, decision.sandbox, decision.requestId, {
        reason: decision.reason,
        forget: decision.forget,
      });
    case 'forget':
      return forgetNetworkRule(workspace, agent, decision.rule);
  }
};

/** Every decision answers with the agent's updated policy, which replaces the cached one. */
export const useDecideNetworkRequest = (workspace: string, agent: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (decision: NetworkDecision) => decide(workspace, agent, decision),
    onSuccess: (policy) => {
      queryClient.setQueryData(networkPolicyQueryKey(workspace, agent), policy);
    },
  });
};
