// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { useAgentsListDeployments } from '@nemo/sdk/generated/agents/agent-deployments';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import type { AgentsListDeploymentsParams } from '@nemo/sdk/generated/agents/schema/AgentsListDeploymentsParams';
import { bareName } from '@studio/components/evaluation/submitEvaluationJob';
import { useMemo } from 'react';

interface UseRunningDeploymentsParams {
  workspace: string;
  agent?: string;
}

interface UseRunningDeploymentsResult {
  deployments: AgentDeployment[];
  isLoading: boolean;
}

/** Polled so a deployment that stops while the wizard is open drops out of the list. */
export const useRunningDeployments = ({
  workspace,
  agent,
}: UseRunningDeploymentsParams): UseRunningDeploymentsResult => {
  const agentName = agent ? bareName(agent) : undefined;
  // Filtered server-side: the list is workspace-wide and paged, so an unfiltered first page can
  // miss the agent's deployments and the wizard would refuse to run. The generated params type
  // omits `filter` because the API declares it through a dependency, not query parameters.
  const params: AgentsListDeploymentsParams & {
    filter: { agent?: string; status: 'running' };
  } = { page_size: 100, filter: { agent: agentName, status: 'running' } };
  const { data, isLoading } = useAgentsListDeployments(workspace, params, {
    query: { enabled: !!agentName, refetchInterval: JOB_POLLING_INTERVAL_LONG },
  });

  const deployments = useMemo(
    () =>
      (data?.data ?? []).filter(
        (deployment) =>
          deployment.agent === agentName && deployment.status === 'running' && !!deployment.name
      ),
    [data, agentName]
  );

  return { deployments, isLoading };
};
