// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import {
  Badge,
  Flex,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { hasAgentConfig } from '@studio/api/agents/hasAgentConfig';
import { FABRIC_CONFIG_FORMAT } from '@studio/api/agents/packageAgent';
import { agentSpecSource, useAgentSpecFileset } from '@studio/api/agents/useAgentSpecFileset';
import { getAgentModelNames } from '@studio/components/dataViews/AgentsDataView/utils';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { ImportTracesModal } from '@studio/components/ImportTracesModal';
import {
  AGENT_CONTAINER_DEPLOYMENTS_ENABLED,
  AGENT_OPTIMIZATION_FORM_ENABLED,
  AGENT_OPTIMIZATIONS_ENABLED,
  AGENT_OVERVIEW_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentDeploymentsListRoute/CreateDeploymentModal';
import { AgentDetailCTAs } from '@studio/routes/agents/AgentDetailRoute/AgentDetailCTAs';
import {
  BuildThenDeploy,
  type PendingImageBuild,
} from '@studio/routes/agents/AgentDetailRoute/BuildThenDeploy';
import { ChatPlaygroundContent } from '@studio/routes/agents/AgentDetailRoute/ChatPlaygroundContent';
import { DeploymentLogsView } from '@studio/routes/agents/AgentDetailRoute/DeploymentLogsView';
import { DeploymentsTab } from '@studio/routes/agents/AgentDetailRoute/DeploymentsTab';
import { DetailsTab } from '@studio/routes/agents/AgentDetailRoute/DetailsTab';
import { EvaluationsTab } from '@studio/routes/agents/AgentDetailRoute/EvaluationsTab';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { InsightsTab } from '@studio/routes/agents/AgentDetailRoute/InsightsTab';
import { LaunchOptimizeModal } from '@studio/routes/agents/AgentDetailRoute/optimizations/LaunchOptimizeModal';
import { OptimizationsTab } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationsTab';
import { OverviewTab } from '@studio/routes/agents/AgentDetailRoute/OverviewTab';
import { SOURCE_PANEL_ID } from '@studio/routes/agents/AgentDetailRoute/SourcePanel';
import {
  ACTION_SEARCH_PARAM,
  type AgentDetailTab,
  DEFAULT_TAB,
  isAgentDetailTab,
  TAB_SEARCH_PARAM,
} from '@studio/routes/agents/AgentDetailRoute/tabs';
import { useAgentDetails } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { deriveWalkthroughStep } from '@studio/routes/agents/AgentDetailRoute/walkthrough';
import { WalkthroughCoachmarks } from '@studio/routes/agents/AgentDetailRoute/WalkthroughCoachmarks';
import {
  clearAgentWalkthroughPending,
  isAgentWalkthroughPending,
} from '@studio/routes/agents/AgentDetailRoute/walkthroughStorage';
import { getAgentsListRoute } from '@studio/routes/utils';
import { GitCommitHorizontal } from 'lucide-react';
import { type FC, useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router';

const VIEW_SEARCH_PARAM = 'view';
const VIEW_NEW = 'new';

export const AgentDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { [ROUTE_PARAMS.agentName]: agentName } = useParams<{ agentName: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedDeploymentName, setSelectedDeploymentName] = useState<string | undefined>();
  const [logsDeploymentName, setLogsDeploymentName] = useState<string | undefined>();
  const [createDeploymentOpen, setCreateDeploymentOpen] = useState(false);
  const [submitEvalOpen, setSubmitEvalOpen] = useState(false);
  const [importTracesOpen, setImportTracesOpen] = useState(false);
  const [launchOptimizeOpen, setLaunchOptimizeOpen] = useState(false);
  const [deleteDeploymentTarget, setDeleteDeploymentTarget] = useState<AgentDeployment | null>(
    null
  );
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const deployButtonRef = useRef<HTMLDivElement>(null);
  const tabsRef = useRef<HTMLDivElement>(null);
  const [walkthroughActive, setWalkthroughActive] = useState(false);
  const [walkthroughDismissed, setWalkthroughDismissed] = useState(false);
  const tabFromUrl = searchParams.get(TAB_SEARCH_PARAM);
  const selectedTab: AgentDetailTab = isAgentDetailTab(tabFromUrl) ? tabFromUrl : DEFAULT_TAB;
  const isCreatingOptimization =
    AGENT_OPTIMIZATION_FORM_ENABLED &&
    selectedTab === 'optimizations' &&
    searchParams.get(VIEW_SEARCH_PARAM) === VIEW_NEW;

  const {
    agent,
    isAgentLoading,
    isAgentPending,
    agentDeployments,
    agentEvals,
    isAgentEvalsPending,
    agentJobs,
    chatDeployment,
    deleteDeploymentMutation,
    healthyDeployments,
    isDeploying,
    isDeploymentsLoading,
  } = useAgentDetails({ workspace, agentName, selectedDeploymentName });
  const { data: specFileset } = useAgentSpecFileset(workspace, agentName);
  const specSource = agentSpecSource(specFileset);

  useBreadcrumbs({
    items: [
      { slotLabel: 'Agents', href: getAgentsListRoute(workspace) },
      { slotLabel: agentName ?? 'Agent details' },
    ],
  });

  useEffect(() => {
    setWalkthroughDismissed(false);
    setWalkthroughActive(!!agentName && isAgentWalkthroughPending(agentName));
  }, [agentName]);

  const walkthroughStep = deriveWalkthroughStep({
    active: walkthroughActive,
    dismissed: walkthroughDismissed,
    createDeploymentOpen,
    selectedTab,
    hasDeployment: agentDeployments.length > 0,
    hasHealthyDeployment: healthyDeployments.length > 0,
  });

  const endWalkthrough = () => {
    setWalkthroughDismissed(true);
    if (agentName) clearAgentWalkthroughPending(agentName);
  };

  const setSelectedTab = (tab: AgentDetailTab) => {
    setSearchParams({ [TAB_SEARCH_PARAM]: tab }, { replace: true });
  };

  const setOptimizationView = (creating: boolean) => {
    const params = new URLSearchParams({ [TAB_SEARCH_PARAM]: 'optimizations' });
    if (creating) params.set(VIEW_SEARCH_PARAM, VIEW_NEW);
    setSearchParams(params);
  };

  // The in-tab form is still behind its own flag; until it ships, Optimize opens the launch modal.
  const openOptimize = () =>
    AGENT_OPTIMIZATION_FORM_ENABLED ? setOptimizationView(true) : setLaunchOptimizeOpen(true);

  const switchToChat = (deployment: AgentDeployment) => {
    setSelectedDeploymentName(deployment.name);
    setSelectedTab('chat');
  };

  const viewLogs = (deployment: AgentDeployment) => {
    setLogsDeploymentName(deployment.name);
    setSelectedTab('logs');
  };

  const modelNames = getAgentModelNames(agent?.config);
  const canDeploy = hasAgentConfig(agent?.config);
  // Narrower than canDeploy: NAT workflows package from a source checkout.
  const canPackage = agent?.config_format === FABRIC_CONFIG_FORMAT;
  // Survives closing the deploy modal, but not a change of agent: the route is
  // reused across agentName, so an unscoped tag would deploy one agent's image
  // under another's name.
  const [builtImage, setBuiltImage] = useState<{ agent: string; image: string } | undefined>();
  const builtImageForAgent = builtImage?.agent === agentName ? builtImage?.image : undefined;
  // Stable identity, and a no-op when nothing changed: the panel reports the tag
  // from an effect keyed on this callback, so a new closure or a new object here
  // re-runs it forever.
  const rememberBuiltImage = useCallback(
    (image: string) => {
      setBuiltImage((current) =>
        current?.agent === agentName && current?.image === image
          ? current
          : { agent: agentName ?? '', image }
      );
    },
    [agentName]
  );

  // Scoped like builtImage: the route is reused across agents.
  const [pendingBuild, setPendingBuild] = useState<
    { agent: string; build: PendingImageBuild } | undefined
  >();
  const pendingBuildForAgent = pendingBuild?.agent === agentName ? pendingBuild?.build : undefined;
  const reportPendingBuild = useCallback(
    (build: PendingImageBuild | null) =>
      setPendingBuild(build && agentName ? { agent: agentName, build } : undefined),
    [agentName]
  );

  const canRunEvaluation = !!agentName && canDeploy;

  const actionFromUrl = searchParams.get(ACTION_SEARCH_PARAM);
  // Waits for the agent query to settle so both modals are gated on a real agent, then strips the
  // param either way: a link that cannot open its modal still lands on its tab. `isPending`, not
  // `isLoading`: a paused retry (hidden tab, offline) is not loading, but has not settled either.
  useEffect(() => {
    if (!actionFromUrl || isAgentPending) return;
    if (actionFromUrl === 'run-evaluation' && canRunEvaluation) setSubmitEvalOpen(true);
    const optimizeRequested =
      actionFromUrl === 'optimize' && AGENT_OPTIMIZATIONS_ENABLED && !!agent;
    if (optimizeRequested && !AGENT_OPTIMIZATION_FORM_ENABLED) setLaunchOptimizeOpen(true);
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current);
        next.delete(ACTION_SEARCH_PARAM);
        if (optimizeRequested && AGENT_OPTIMIZATION_FORM_ENABLED) {
          next.set(TAB_SEARCH_PARAM, 'optimizations');
          next.set(VIEW_SEARCH_PARAM, VIEW_NEW);
        }
        return next;
      },
      { replace: true }
    );
  }, [actionFromUrl, isAgentPending, canRunEvaluation, agent, setSearchParams]);

  const status = healthyDeployments.length > 0 ? 'running' : agentDeployments[0]?.status;
  const statusPillLabel =
    healthyDeployments.length > 0
      ? 'Healthy'
      : status === 'pending' || status === 'starting'
        ? 'Deploying'
        : status === 'deleting'
          ? 'Deleting'
          : status === 'failed'
            ? 'Failed'
            : agentDeployments.length === 0
              ? 'No deployments'
              : (status ?? 'Unknown');

  return (
    <AccessibleTitle title={`${agentName ?? 'Agent'} details for ${workspace}`}>
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="shrink-0 p-0"
          slotHeading={
            <Flex align="baseline" gap="3">
              <Text kind="title/md">{agent?.name ?? agentName ?? 'Agent details'}</Text>
              <StatusBadge status={status} label={statusPillLabel} />
              {specSource ? (
                <Link
                  to={{ search: `?${TAB_SEARCH_PARAM}=details`, hash: `#${SOURCE_PANEL_ID}` }}
                  className="contents"
                  aria-label={`Source: ${specSource.repository} at ${specSource.revision}`}
                >
                  <Badge kind="solid" color="gray" className="cursor-pointer">
                    <GitCommitHorizontal size={12} aria-hidden />
                    {shortRevision(specSource.revision)}
                  </Badge>
                </Link>
              ) : null}
            </Flex>
          }
          slotActions={
            <AgentDetailCTAs
              tab={selectedTab}
              agentName={agentName}
              canDeploy={canDeploy}
              canRunEvaluation={canRunEvaluation}
              isDeploying={isDeploying}
              canOptimize={!isCreatingOptimization}
              deployButtonRef={deployButtonRef}
              onDeploy={() => setCreateDeploymentOpen(true)}
              onRunEvaluation={() => setSubmitEvalOpen(true)}
              onOptimize={openOptimize}
              onImportTraces={() => setImportTracesOpen(true)}
            />
          }
        ></PageHeader>

        <TabsRoot
          className="flex min-h-0 flex-1 flex-col"
          value={selectedTab}
          onValueChange={(value) => {
            if (isAgentDetailTab(value)) setSelectedTab(value);
          }}
        >
          <TabsList className="shrink-0" ref={tabsRef}>
            {AGENT_OVERVIEW_ENABLED && <TabsTrigger value="overview">Overview</TabsTrigger>}
            <TabsTrigger value="deployments">Deployments</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
            <TabsTrigger value="chat">Chat</TabsTrigger>
            <TabsTrigger value="evaluations">Evaluations</TabsTrigger>
            {AGENT_OPTIMIZATIONS_ENABLED && (
              <TabsTrigger value="optimizations">Optimizations</TabsTrigger>
            )}
            {OPTIMIZER_ENABLED && <TabsTrigger value="insights">Insights</TabsTrigger>}
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>

          {AGENT_OVERVIEW_ENABLED && (
            <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="overview">
              <OverviewTab
                workspace={workspace}
                agent={agent}
                modelNames={modelNames}
                evals={agentEvals}
                isEvalsPending={isAgentEvalsPending}
                onRunAgent={() => setSelectedTab('chat')}
                onRunEvaluation={canRunEvaluation ? () => setSubmitEvalOpen(true) : undefined}
              />
            </TabsContent>
          )}

          <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="evaluations">
            <EvaluationsTab
              workspace={workspace}
              agentName={agentName}
              evals={agentEvals}
              jobs={agentJobs}
            />
          </TabsContent>

          {AGENT_OPTIMIZATIONS_ENABLED && (
            <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="optimizations">
              <OptimizationsTab
                agentName={agentName}
                evals={agentEvals}
                isEvalsPending={isAgentEvalsPending}
                isCreating={isCreatingOptimization}
                onOptimize={openOptimize}
                onCloseForm={() => setOptimizationView(false)}
              />
            </TabsContent>
          )}

          {OPTIMIZER_ENABLED && (
            <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="insights">
              <InsightsTab workspace={workspace} agentName={agentName} agent={agent} />
            </TabsContent>
          )}

          <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="deployments">
            <DeploymentsTab
              agentName={agentName}
              deployments={agentDeployments}
              isDeploymentsLoading={isDeploymentsLoading}
              isDeploying={isDeploying}
              onDeploy={() => setCreateDeploymentOpen(true)}
              onChat={switchToChat}
              onDelete={setDeleteDeploymentTarget}
              onViewLogs={viewLogs}
              canDeploy={canDeploy}
              specSource={specSource}
              workspace={workspace}
              canPackage={canPackage}
              isAgentLoading={isAgentLoading}
              onImageBuilt={(image) => {
                rememberBuiltImage(image);
                setCreateDeploymentOpen(true);
              }}
              onImageAvailable={rememberBuiltImage}
              pendingBuild={pendingBuildForAgent}
            />
          </TabsContent>

          <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="logs">
            <DeploymentLogsView
              workspace={workspace}
              deployments={agentDeployments}
              selectedDeploymentName={logsDeploymentName}
              onSelectDeployment={setLogsDeploymentName}
            />
          </TabsContent>

          <TabsContent className="min-h-0 flex-1 overflow-hidden p-0 pt-6" value="chat">
            <div className="mx-auto flex h-full w-full max-w-4xl flex-col">
              <ChatPlaygroundContent
                workspace={workspace}
                agentName={agentName}
                chatDeployment={chatDeployment}
                healthyDeployments={healthyDeployments}
                isDeploymentsLoading={isDeploymentsLoading}
                isDeploying={isDeploying}
                chatAreaRef={chatAreaRef}
                onSelectDeployment={setSelectedDeploymentName}
                onDeploy={() => setCreateDeploymentOpen(true)}
                canDeploy={canDeploy}
              />
            </div>
          </TabsContent>

          <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="details">
            <DetailsTab workspace={workspace} agentName={agentName} agent={agent} />
          </TabsContent>
        </TabsRoot>
      </Stack>
      <SubmitEvaluationModal
        open={submitEvalOpen}
        onClose={() => setSubmitEvalOpen(false)}
        workspace={workspace}
        agent={agentName}
      />
      {agentName && importTracesOpen && (
        <ImportTracesModal
          open
          onClose={() => setImportTracesOpen(false)}
          workspace={workspace}
          agent={agentName}
        />
      )}
      {agentName && launchOptimizeOpen && (
        <LaunchOptimizeModal
          open
          onClose={() => setLaunchOptimizeOpen(false)}
          workspace={workspace}
          agentName={agentName}
        />
      )}
      {agentName && AGENT_CONTAINER_DEPLOYMENTS_ENABLED ? (
        <BuildThenDeploy
          key={agentName}
          workspace={workspace}
          agentName={agentName}
          onPendingChange={reportPendingBuild}
        />
      ) : null}
      {createDeploymentOpen && (
        <CreateDeploymentModal
          open
          agent={agentName}
          workspace={workspace}
          initialImage={builtImageForAgent}
          onClose={() => setCreateDeploymentOpen(false)}
        />
      )}
      <WalkthroughCoachmarks
        walkthroughStep={walkthroughStep}
        deployButtonRef={deployButtonRef}
        tabsRef={tabsRef}
        chatAreaRef={chatAreaRef}
        onDismiss={endWalkthrough}
      />
      {deleteDeploymentTarget && (
        <DeleteConfirmationModal
          open
          title="Delete Deployment"
          successText="Successfully queued deployment for deletion."
          onDelete={async () => {
            try {
              if (!deleteDeploymentTarget.name) return false;
              await deleteDeploymentMutation.mutateAsync({
                workspace,
                name: deleteDeploymentTarget.name,
              });
              return true;
            } catch {
              return false;
            }
          }}
          onClose={() => setDeleteDeploymentTarget(null)}
          simpleConfirm
        />
      )}
    </AccessibleTitle>
  );
};
