// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { IconError } from '@nemo/icons/icons';
import { Button, PageHeader, Stack, Text } from '@nvidia/foundations-react-core';
import { AgentsTable, type AgentTableRow } from '@studio/components/dataViews/AgentsDataView';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  usePluginInstalled,
  usePluginsError,
  usePluginsLoaded,
} from '@studio/plugins/PluginContext';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CloneAgentModal } from '@studio/routes/agents/AgentsListRoute/CloneAgentModal';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentsListRoute/CreateDeploymentModal';
import { NewAgentModal } from '@studio/routes/agents/AgentsListRoute/NewAgentModal';
import { isNewAgentTab } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/const';
import type { NewAgentTab } from '@studio/routes/agents/AgentsListRoute/NewAgentModal/type';
import { getAgentDetailRoute, REGISTER_AGENT_SEARCH_PARAM } from '@studio/routes/utils';
import { type FC, useEffect, useState } from 'react';
import { useSearchParams } from 'react-router';

export const AgentsListRoute: FC = () => {
  const pluginsLoaded = usePluginsLoaded();
  const pluginsError = usePluginsError();
  const agentsInstalled = usePluginInstalled('agents');
  const workspace = useWorkspaceFromPath();
  const [searchParams, setSearchParams] = useSearchParams();
  const openRow = useRowNavigation();
  const [createDeploymentAgent, setCreateDeploymentAgent] = useState<string | null>(null);
  const [newAgentTab, setNewAgentTab] = useState<NewAgentTab | null>(null);
  const [cloneSource, setCloneSource] = useState<AgentTableRow | null>(null);

  const requestedTab = searchParams.get(REGISTER_AGENT_SEARCH_PARAM);
  useEffect(() => {
    if (requestedTab === null) {
      return;
    }
    if (isNewAgentTab(requestedTab)) {
      setNewAgentTab(requestedTab);
    }
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete(REGISTER_AGENT_SEARCH_PARAM);
        return next;
      },
      { replace: true }
    );
  }, [requestedTab, setSearchParams]);

  useBreadcrumbs({
    items: [{ slotLabel: 'Agents' }],
  });

  const handleOpenDetails = (agent: AgentTableRow, event?: React.MouseEvent) =>
    openRow(event, getAgentDetailRoute(workspace, agent.name));

  if (!pluginsLoaded && !pluginsError) {
    return null;
  }

  if (pluginsLoaded && !pluginsError && !agentsInstalled) {
    return (
      <Stack className="h-full justify-center mx-auto max-w-[640px]" gap="density-md">
        <IconError className="size-16 stroke-2" color="var(--text-color-feedback-danger)" />
        <Text kind="display/xl">Plugin Not Enabled</Text>
        <Text kind="title/lg">The Agents plugin is not installed.</Text>
        <Text lineHeight="150">
          To use this page, the <strong>agents</strong> plugin must be registered with the platform.
          Ask your administrator to install and enable the agents plugin.
        </Text>
      </Stack>
    );
  }

  return (
    <AccessibleTitle title={`Agents for ${workspace}`}>
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Agents"
          slotDescription="View and manage AI agents and their deployments."
          slotActions={
            <Button color="brand" onClick={() => setNewAgentTab('upload')}>
              Register Agent
            </Button>
          }
        />
        <AgentsTable
          onAgentRowClick={handleOpenDetails}
          onCreateDeployment={(agentName) => setCreateDeploymentAgent(agentName)}
          onCloneAgent={setCloneSource}
        />
      </Stack>
      <NewAgentModal
        open={newAgentTab !== null}
        onClose={() => setNewAgentTab(null)}
        workspace={workspace}
        initialTab={newAgentTab ?? 'upload'}
      />
      <CloneAgentModal
        open={cloneSource !== null}
        onClose={() => setCloneSource(null)}
        workspace={workspace}
        sourceAgent={cloneSource}
      />
      <CreateDeploymentModal
        open={createDeploymentAgent !== null}
        agent={createDeploymentAgent || undefined}
        onClose={() => setCreateDeploymentAgent(null)}
        workspace={workspace}
      />
    </AccessibleTitle>
  );
};
