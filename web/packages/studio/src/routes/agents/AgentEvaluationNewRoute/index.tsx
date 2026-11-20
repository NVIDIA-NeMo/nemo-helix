// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import { hasAgentConfig } from '@studio/api/agents/hasAgentConfig';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { RerunEvaluationForm } from '@studio/components/evaluation/AgentEvaluationWizard/RerunEvaluationForm';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { getAgentDetailRoute, getAgentsListRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Navigate, useParams, useSearchParams } from 'react-router';

/** `/workspaces/:workspace/agents/:agentName/evaluations/new[?job=<job to re-run>]` */
export const AgentEvaluationNewRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { [ROUTE_PARAMS.agentName]: agentName = '' } = useParams<{ agentName: string }>();
  const [searchParams] = useSearchParams();
  const sourceJob = searchParams.get('job') ?? undefined;
  const { data: agent, isPending } = useAgentsGetAgent(workspace, agentName, {
    query: { enabled: !!agentName },
  });

  useBreadcrumbs({
    items: [
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      { slotLabel: agentName, href: getAgentDetailRoute(workspace, agentName) },
      { slotLabel: sourceJob ? 'Re-run Evaluation' : 'New Evaluation' },
    ],
  });

  if (isPending) return <Loading description="Loading agent..." />;
  // Same gate as the agent page's Run Evaluation button: an agent without a config cannot run.
  if (!hasAgentConfig(agent?.config)) {
    return <Navigate to={getAgentDetailRoute(workspace, agentName)} replace />;
  }

  return sourceJob ? (
    <RerunEvaluationForm
      key={`${agentName}/${sourceJob}`}
      workspace={workspace}
      agent={agentName}
      sourceJob={sourceJob}
    />
  ) : (
    <AgentEvaluationWizard key={agentName} workspace={workspace} agent={agentName} />
  );
};
