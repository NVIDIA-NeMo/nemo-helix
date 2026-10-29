// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
import {
  agentsListDeployments,
  getAgentsListDeploymentsQueryKey,
} from '@nemo/sdk/generated/agents/agent-deployments';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentsListDeploymentsParams } from '@nemo/sdk/generated/agents/schema/AgentsListDeploymentsParams';
import { skipToken, useQuery } from '@tanstack/react-query';

// The API accepts `filter`, but the generated params type omits it.
type AgentFilteredParams = AgentsListDeploymentsParams & { filter: { agent: string } };

const NO_DEPLOYMENTS: AgentDeployment[] = [];

interface UseAgentDeploymentsParams {
  workspace: string;
  agent?: string;
  refetchInterval: number | ((deployments: AgentDeployment[]) => number);
}

export const useAgentDeployments = ({
  workspace,
  agent,
  refetchInterval,
}: UseAgentDeploymentsParams) => {
  const { data, isLoading } = useQuery({
    queryKey: [...getAgentsListDeploymentsQueryKey(workspace), { agent }],
    queryFn: agent
      ? ({ signal }) =>
          fetchAllPages((page, pageSize) => {
            const params: AgentFilteredParams = { page, page_size: pageSize, filter: { agent } };
            return agentsListDeployments(workspace, params, signal);
          })
      : skipToken,
    refetchInterval: (query) =>
      typeof refetchInterval === 'number'
        ? refetchInterval
        : refetchInterval(query.state.data ?? NO_DEPLOYMENTS),
  });

  return { deployments: data ?? NO_DEPLOYMENTS, isLoading };
};
