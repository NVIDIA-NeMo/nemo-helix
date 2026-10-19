// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListDeploymentsQueryKey,
  useAgentsCreateDeployment,
  useAgentsListDeploymentModes,
} from '@nemo/sdk/generated/agents/agent-deployments';
import { buildThenDeployNavigation } from '@studio/api/agents/buildThenDeploy';
import {
  type DeploymentMode,
  DeploymentModeAvailabilityMode,
  IMAGE_DEPLOYMENT_MODES,
  useDeploymentModes,
} from '@studio/api/agents/useDeploymentModes';
import { useImageBuildsUnsupported } from '@studio/api/agents/useImageBuildsUnsupported';
import { AGENT_CONTAINER_DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router';
import { z } from 'zod';

const OFFERED_ON_CREATE: readonly DeploymentMode[] = AGENT_CONTAINER_DEPLOYMENTS_ENABLED
  ? ['subprocess', ...IMAGE_DEPLOYMENT_MODES]
  : ['subprocess'];

/** Form fields for a create flow that can also deploy; merge into the flow's own schema. */
export const deployOnCreateFormSchema = z.object({
  deploy: z.boolean().default(true),
  deploymentMode: z.nativeEnum(DeploymentModeAvailabilityMode).default('subprocess'),
});

export type DeployOnCreateFormData = z.infer<typeof deployOnCreateFormSchema>;

export const DEPLOY_ON_CREATE_FORM_DEFAULTS: DeployOnCreateFormData =
  deployOnCreateFormSchema.parse({});

export interface DeployOnCreatePlan {
  mode: DeploymentMode;
  buildImage: boolean;
}

export interface DeployOnCreate {
  /** Runtimes a brand-new agent can be deployed to. Empty while loading or when none can be. */
  modes: readonly DeploymentMode[];
  isLoading: boolean;
  canDeploy: boolean;
  /** Why the runtimes could not be read, or null. A requested deployment waits on a retry or opt-out. */
  discoveryError: string | null;
  isRetrying: boolean;
  retry: () => void;
  /** Whether submit must wait: a requested deployment cannot be planned until the runtimes are known. */
  blocksSubmit: (deploy: boolean) => boolean;
  needsImageBuild: (mode: DeploymentMode) => boolean;
  /** The deployment to run once the agent exists, or null to only create it. */
  plan: (choice: DeployOnCreateFormData) => DeployOnCreatePlan | null;
  /** Whether a deployment request is in flight; callers hold submit until it settles. */
  isDeploying: boolean;
  /** Opens the new agent's page, deploying it first or handing an image build to that page. */
  openCreatedAgent: (
    workspace: string,
    agentName: string,
    plan: DeployOnCreatePlan | null
  ) => Promise<void>;
}

export const useDeployOnCreate = (
  workspace: string,
  { enabled }: { enabled: boolean }
): DeployOnCreate => {
  const toast = useToast();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // A new agent has no image yet, so a mode that needs one is offered only if the platform can build it.
  const modeState = useDeploymentModes(workspace, { enabled });
  const imageBuildsUnsupported = useImageBuildsUnsupported();
  const modes = useMemo<readonly DeploymentMode[]>(
    () =>
      modeState.status === 'ready'
        ? OFFERED_ON_CREATE.filter(
            (mode) =>
              modeState.withoutImage.includes(mode) ||
              (modeState.enabled.includes(mode) && !imageBuildsUnsupported)
          )
        : [],
    [modeState, imageBuildsUnsupported]
  );
  const needsImageBuild = useCallback(
    (mode: DeploymentMode) =>
      modeState.status === 'ready' && !modeState.withoutImage.includes(mode),
    [modeState]
  );
  const canDeploy = modes.length > 0;
  const isLoading = modeState.status === 'loading';

  // Shares the modes query's cache entry; read here for its error and a retry.
  const {
    error: modesError,
    isFetching: isRetrying,
    refetch,
  } = useAgentsListDeploymentModes(workspace, { query: { enabled } });
  let discoveryError: string | null = null;
  if (modeState.status === 'unknown') {
    discoveryError = (modesError && getErrorMessage(modesError)) || 'unknown error';
  }
  const retry = useCallback(() => void refetch(), [refetch]);
  const blocksSubmit = useCallback(
    (deploy: boolean) => deploy && (isLoading || discoveryError !== null),
    [isLoading, discoveryError]
  );

  // Toasts live on the hook, not on mutate(): opening the new agent unmounts the caller.
  const { mutateAsync: deployAgent, isPending: isDeploying } = useAgentsCreateDeployment({
    mutation: {
      onSuccess: (deployment, { workspace: deploymentWorkspace }) => {
        toast.success(`Deploying agent "${deployment.agent}"`);
        void queryClient.invalidateQueries({
          queryKey: getAgentsListDeploymentsQueryKey(deploymentWorkspace),
        });
      },
      onError: (error, { data }) => {
        toast.error(
          `Agent "${data.agent}" was created, but deploying it failed: ${getErrorMessage(error) || 'unknown error'}`
        );
      },
    },
  });

  const plan = useCallback(
    ({ deploy, deploymentMode }: DeployOnCreateFormData): DeployOnCreatePlan | null => {
      if (!deploy || !canDeploy) return null;
      const mode = modes.includes(deploymentMode) ? deploymentMode : modes[0]!;
      return { mode, buildImage: needsImageBuild(mode) };
    },
    [canDeploy, modes, needsImageBuild]
  );

  const openCreatedAgent = useCallback(
    async (agentWorkspace: string, agentName: string, deployment: DeployOnCreatePlan | null) => {
      if (deployment?.buildImage) {
        // The build outlives the caller, so the agent's page runs it and deploys the tag.
        navigate(
          ...buildThenDeployNavigation(agentWorkspace, agentName, { mode: deployment.mode })
        );
        return;
      }
      if (deployment) {
        // A failure is toasted by onError; the agent exists either way, so its page still opens.
        await deployAgent({
          workspace: agentWorkspace,
          data: { agent: agentName, deployment_mode: deployment.mode },
        }).catch(() => undefined);
      }
      navigate(getAgentDetailRoute(agentWorkspace, agentName));
    },
    [navigate, deployAgent]
  );

  return {
    modes,
    isLoading,
    canDeploy,
    discoveryError,
    isRetrying,
    retry,
    blocksSubmit,
    needsImageBuild,
    plan,
    isDeploying,
    openCreatedAgent,
  };
};
