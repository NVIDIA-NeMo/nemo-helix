// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Cog, Commit, CopyDoc } from '@nemo/icons/icons';
import type { OptimizationStrategy } from '@nemo/sdk/generated/agent-optimization/schema/OptimizationStrategy';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import {
  Badge,
  Button,
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
import { AgentConfigPanel } from '@studio/routes/agents/AgentDetailRoute/AgentConfigPanel';
import { AgentDetailCTAs } from '@studio/routes/agents/AgentDetailRoute/AgentDetailCTAs';
import {
  BuildThenDeploy,
  type PendingImageBuild,
} from '@studio/routes/agents/AgentDetailRoute/BuildThenDeploy';
import { ChatPlaygroundContent } from '@studio/routes/agents/AgentDetailRoute/ChatPlaygroundContent';
import { DeploymentLogsView } from '@studio/routes/agents/AgentDetailRoute/DeploymentLogsView';
import { DeploymentsTab } from '@studio/routes/agents/AgentDetailRoute/DeploymentsTab';
import { EvaluationsTab } from '@studio/routes/agents/AgentDetailRoute/EvaluationsTab';
import { HeaderDeploymentSelect } from '@studio/routes/agents/AgentDetailRoute/HeaderDeploymentSelect';
import { shortRevision } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { InsightsTab } from '@studio/routes/agents/AgentDetailRoute/InsightsTab';
import { LaunchOptimizeModal } from '@studio/routes/agents/AgentDetailRoute/optimizations/LaunchOptimizeModal';
import { OptimizationsTab } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationsTab';
import { OverviewTab } from '@studio/routes/agents/AgentDetailRoute/OverviewTab';
import {
  ACTION_SEARCH_PARAM,
  type AgentDetailTab,
  DEFAULT_TAB,
  isAgentDetailTab,
  isOptimizationView,
  OptimizationView,
  TAB_SEARCH_PARAM,
  VIEW_SEARCH_PARAM,
} from '@studio/routes/agents/AgentDetailRoute/tabs';
import { useAgentDetails } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { deriveWalkthroughStep } from '@studio/routes/agents/AgentDetailRoute/walkthrough';
import { WalkthroughCoachmarks } from '@studio/routes/agents/AgentDetailRoute/WalkthroughCoachmarks';
import {
  clearAgentWalkthroughPending,
  isAgentWalkthroughPending,
} from '@studio/routes/agents/AgentDetailRoute/walkthroughStorage';
import { CreateDeploymentModal } from '@studio/routes/agents/AgentsListRoute/CreateDeploymentModal';
import { NewAgentModal } from '@studio/routes/agents/AgentsListRoute/NewAgentModal';
import { getAgentRunEvaluationRoute, getAgentsListRoute } from '@studio/routes/utils';
import { type FC, useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router';

export const AgentDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { [ROUTE_PARAMS.agentName]: agentName } = useParams<{ agentName: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedDeploymentName, setSelectedDeploymentName] = useState<string | undefined>();
  const [createDeploymentOpen, setCreateDeploymentOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [importTracesOpen, setImportTracesOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [launchOptimizeOpen, setLaunchOptimizeOpen] = useState(false);
  const [launchStrategy, setLaunchStrategy] = useState<OptimizationStrategy | undefined>();
  const [deleteDeploymentTarget, setDeleteDeploymentTarget] = useState<AgentDeployment | null>(
    null
  );
  const { success, error } = useToast();
  const chatAreaRef = useRef<HTMLDivElement>(null);
  const deployButtonRef = useRef<HTMLDivElement>(null);
  const tabsRef = useRef<HTMLDivElement>(null);
  const [walkthroughActive, setWalkthroughActive] = useState(false);
  const [walkthroughDismissed, setWalkthroughDismissed] = useState(false);
  const tabFromUrl = searchParams.get(TAB_SEARCH_PARAM);
  const selectedTab: AgentDetailTab = isAgentDetailTab(tabFromUrl) ? tabFromUrl : DEFAULT_TAB;
  const viewFromUrl = searchParams.get(VIEW_SEARCH_PARAM);
  // Without the form flag there is nothing to create in the tab, so it only ever shows the table.
  const optimizationView: OptimizationView =
    AGENT_OPTIMIZATION_FORM_ENABLED && isOptimizationView(viewFromUrl)
      ? viewFromUrl
      : OptimizationView.Table;
  const isCreatingOptimization =
    selectedTab === 'optimizations' && optimizationView !== OptimizationView.Table;

  const {
    agent,
    isAgentLoading,
    isAgentPending,
    agentDeployments,
    agentEvals,
    isAgentEvalsPending,
    agentJobs,
    selectedDeployment,
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

  const setOptimizationView = (view: OptimizationView) => {
    const params = new URLSearchParams({ [TAB_SEARCH_PARAM]: 'optimizations' });
    if (view !== OptimizationView.Table) params.set(VIEW_SEARCH_PARAM, view);
    setSearchParams(params);
  };

  // The in-tab flow is still behind its own flag; until it ships, Optimize opens the launch modal.
  const openOptimize = () =>
    AGENT_OPTIMIZATION_FORM_ENABLED
      ? setOptimizationView(OptimizationView.Strategy)
      : setLaunchOptimizeOpen(true);

  const switchToChat = (deployment: AgentDeployment) => {
    setSelectedDeploymentName(deployment.name);
    setSelectedTab('chat');
  };

  const viewLogs = (deployment: AgentDeployment) => {
    setSelectedDeploymentName(deployment.name);
    setSelectedTab('logs');
  };

  const modelNames = getAgentModelNames(agent?.config);
  const canDeploy = hasAgentConfig(agent?.config);
  // An agent with no config has nothing to deploy; the same buttons upload one instead.
  const onDeploy = () => (canDeploy ? setCreateDeploymentOpen(true) : setUploadOpen(true));
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
  const navigate = useNavigate();
  const runEvaluation = () => {
    if (agentName) navigate(getAgentRunEvaluationRoute(workspace, agentName));
  };

  const actionFromUrl = searchParams.get(ACTION_SEARCH_PARAM);
  // Waits for the agent query to settle so the modal is gated on a real agent, then strips the
  // param either way: a link that cannot open its modal still lands on its tab. `isPending`, not
  // `isLoading`: a paused retry (hidden tab, offline) is not loading, but has not settled either.
  useEffect(() => {
    if (!actionFromUrl || isAgentPending) return;
    if (actionFromUrl === 'run-evaluation' && canRunEvaluation && agentName) {
      navigate(getAgentRunEvaluationRoute(workspace, agentName), { replace: true });
      return;
    }
    const optimizeRequested =
      actionFromUrl === 'optimize' && AGENT_OPTIMIZATIONS_ENABLED && !!agent;
    if (optimizeRequested && !AGENT_OPTIMIZATION_FORM_ENABLED) setLaunchOptimizeOpen(true);
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current);
        next.delete(ACTION_SEARCH_PARAM);
        if (optimizeRequested && AGENT_OPTIMIZATION_FORM_ENABLED) {
          next.set(TAB_SEARCH_PARAM, 'optimizations');
          next.set(VIEW_SEARCH_PARAM, 'strategy');
        }
        return next;
      },
      { replace: true }
    );
  }, [
    actionFromUrl,
    isAgentPending,
    agent,
    setSearchParams,
    canRunEvaluation,
    agentName,
    workspace,
    navigate,
  ]);

  const displayName = agent?.name ?? agentName ?? 'Agent details';

  const copyAgentId = async () => {
    if (!agent?.id) return;
    try {
      await navigator.clipboard.writeText(agent.id);
      success('Copied agent ID', { durationMs: 3000 });
    } catch {
      error('Error copying agent ID', { durationMs: 3000 });
    }
  };

  return (
    <AccessibleTitle title={`${agentName ?? 'Agent'} details for ${workspace}`}>
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="shrink-0 p-0"
          slotHeading={
            <Flex align="center" gap="3" className="min-w-0">
              <Text kind="title/md" className="min-w-0 truncate font-bold" title={displayName}>
                {displayName}
              </Text>
              <Flex align="center" gap="3" className="shrink-0">
                <HeaderDeploymentSelect
                  deployments={agentDeployments}
                  selected={selectedDeployment}
                  canCreate={!!agentName && !isDeploying && !isAgentPending}
                  onSelect={setSelectedDeploymentName}
                  onCreate={onDeploy}
                />
                {agent?.id ? (
                  <button
                    type="button"
                    className="group contents"
                    onClick={copyAgentId}
                    aria-label={`Copy agent ID ${agent.id}`}
                  >
                    <Badge kind="solid" color="gray" className="cursor-pointer">
                      <CopyDoc size="12" aria-hidden />
                      ID
                      <span className="max-w-0 overflow-hidden whitespace-nowrap transition-[max-width] duration-200 group-hover:max-w-96 group-focus-visible:max-w-96">
                        {agent.id}
                      </span>
                    </Badge>
                  </button>
                ) : null}
                {specSource ? (
                  <Badge
                    kind="solid"
                    color="gray"
                    aria-label={`Source: ${specSource.repository} at ${specSource.revision}`}
                  >
                    <Commit size="12" aria-hidden />
                    {shortRevision(specSource.revision)}
                  </Badge>
                ) : null}
              </Flex>
            </Flex>
          }
        />

        <TabsRoot
          className="flex min-h-0 flex-1 flex-col"
          value={selectedTab}
          onValueChange={(value) => {
            if (isAgentDetailTab(value)) setSelectedTab(value);
          }}
        >
          <Flex className="shrink-0" align="center" gap="4">
            <TabsList className="min-w-0 flex-1" ref={tabsRef}>
              {AGENT_OVERVIEW_ENABLED && <TabsTrigger value="overview">Overview</TabsTrigger>}
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="evaluations">Evaluations</TabsTrigger>
              {OPTIMIZER_ENABLED && <TabsTrigger value="insights">Insights</TabsTrigger>}
              {AGENT_OPTIMIZATIONS_ENABLED && (
                <TabsTrigger value="optimizations">Optimizations</TabsTrigger>
              )}
              <TabsTrigger value="deployments">Deployment</TabsTrigger>
              <TabsTrigger value="logs">Logs</TabsTrigger>
            </TabsList>
            <Flex align="center" gap="2">
              <Button
                kind="secondary"
                aria-label="Agent config"
                aria-haspopup="dialog"
                onClick={() => setConfigOpen(true)}
              >
                <Cog aria-hidden />
              </Button>
              <AgentDetailCTAs
                tab={selectedTab}
                agentName={agentName}
                canDeploy={canDeploy}
                isAgentPending={isAgentPending}
                canRunEvaluation={canRunEvaluation}
                isDeploying={isDeploying}
                canOptimize={!isCreatingOptimization}
                deployButtonRef={deployButtonRef}
                onDeploy={onDeploy}
                onRunEvaluation={runEvaluation}
                onOptimize={openOptimize}
                onImportTraces={() => setImportTracesOpen(true)}
              />
            </Flex>
          </Flex>

          {AGENT_OVERVIEW_ENABLED && (
            <TabsContent className="min-h-0 flex-1 overflow-auto p-0 pt-6" value="overview">
              <OverviewTab
                workspace={workspace}
                agent={agent}
                modelNames={modelNames}
                evals={agentEvals}
                isEvalsPending={isAgentEvalsPending}
                onRunAgent={() => setSelectedTab('chat')}
                onRunEvaluation={canRunEvaluation ? runEvaluation : undefined}
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
                view={optimizationView}
                onOptimize={openOptimize}
                onViewChange={setOptimizationView}
                onUploadConfig={() => setLaunchOptimizeOpen(true)}
                onRunStrategy={setLaunchStrategy}
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
              onDeploy={onDeploy}
              onChat={switchToChat}
              onDelete={setDeleteDeploymentTarget}
              onViewLogs={viewLogs}
              canDeploy={canDeploy}
              isAgentPending={isAgentPending}
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
            <DeploymentLogsView workspace={workspace} deployment={selectedDeployment} />
          </TabsContent>

          <TabsContent className="min-h-0 flex-1 overflow-hidden p-0 pt-6" value="chat">
            <div className="mx-auto flex h-full w-full max-w-4xl flex-col">
              <ChatPlaygroundContent
                workspace={workspace}
                agentName={agentName}
                deployment={selectedDeployment}
                hasHealthyDeployments={healthyDeployments.length > 0}
                isDeploymentsLoading={isDeploymentsLoading}
                isDeploying={isDeploying}
                chatAreaRef={chatAreaRef}
                onDeploy={onDeploy}
                canDeploy={canDeploy}
                isAgentPending={isAgentPending}
              />
            </div>
          </TabsContent>
        </TabsRoot>
      </Stack>
      <AgentConfigPanel
        open={configOpen}
        workspace={workspace}
        agent={agent}
        specSource={specSource}
        onClose={() => setConfigOpen(false)}
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
      {agentName && launchStrategy && (
        <LaunchOptimizeModal
          key={launchStrategy.name}
          open
          onClose={() => setLaunchStrategy(undefined)}
          workspace={workspace}
          agentName={agentName}
          strategy={launchStrategy}
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
      {uploadOpen && agentName && (
        <NewAgentModal
          open
          workspace={workspace}
          // Names are unique and an agent cannot be replaced, so the upload needs a fresh one.
          initialName={`${agentName}-v2`}
          onClose={() => setUploadOpen(false)}
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
