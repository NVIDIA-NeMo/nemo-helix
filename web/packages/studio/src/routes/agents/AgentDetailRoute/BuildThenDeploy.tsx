// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getAgentsListDeploymentsQueryKey,
  useAgentsCreateDeployment,
} from '@nemo/sdk/generated/agents/agent-deployments';
import {
  type BuildThenDeployRequest,
  getBuildThenDeployRequest,
} from '@studio/api/agents/buildThenDeploy';
import type { DeploymentMode } from '@studio/api/agents/useDeploymentModes';
import { usePackageAgent } from '@studio/api/agents/usePackageAgent';
import { deploymentModeLabel } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router';

export interface PendingImageBuild {
  mode: DeploymentMode;
  jobName?: string;
  isStalled: boolean;
}

interface BuildThenDeployProps {
  workspace: string;
  agentName: string;
  onPendingChange?: (build: PendingImageBuild | null) => void;
}

/**
 * Picks up a build-then-deploy request handed over in the route state, builds the agent's image,
 * and deploys the finished tag. Renders nothing; progress is reported through `onPendingChange`.
 */
export const BuildThenDeploy: FC<BuildThenDeployProps> = ({
  workspace,
  agentName,
  onPendingChange,
}) => {
  const toast = useToast();
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [request, setRequest] = useState<BuildThenDeployRequest | null>(null);
  const handledLocationKey = useRef<string | null>(null);
  const {
    packageAgent,
    submitError,
    jobName,
    isRestored,
    isStalled,
    isComplete,
    isFailed,
    isUnreachable,
    isResultPending,
    resultError,
    image,
  } = usePackageAgent({ workspace, agentName });

  const { mutate: deploy } = useAgentsCreateDeployment({
    mutation: {
      onSuccess: (deployment) => {
        toast.success(`Deploying agent "${deployment.agent}"`);
        void queryClient.invalidateQueries({
          queryKey: getAgentsListDeploymentsQueryKey(workspace),
        });
      },
      onError: (error) => {
        toast.error(
          `The image was built, but deploying it failed: ${getErrorMessage(error) || 'unknown error'}`
        );
      },
    },
  });

  // Keyed on the location so StrictMode's second effect run, and Back, do not start another build.
  useEffect(() => {
    if (handledLocationKey.current === location.key) return;
    handledLocationKey.current = location.key;
    const handedOver = getBuildThenDeployRequest(location.state);
    if (!handedOver) return;
    setRequest(handedOver);
    packageAgent({});
    toast.info(
      `Building an image for "${agentName}". It deploys with ${deploymentModeLabel(handedOver.mode)} when the build finishes.`
    );
    navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
  }, [location, navigate, packageAgent, toast, agentName]);

  // A restored job is an earlier build, never the one this request started.
  const ownJob = Boolean(jobName) && !isRestored;

  const pendingMode = request?.mode;
  const pendingJobName = ownJob ? jobName : undefined;
  const pendingStalled = ownJob && isStalled;
  useEffect(() => {
    onPendingChange?.(
      pendingMode ? { mode: pendingMode, jobName: pendingJobName, isStalled: pendingStalled } : null
    );
  }, [onPendingChange, pendingMode, pendingJobName, pendingStalled]);
  useEffect(() => () => onPendingChange?.(null), [onPendingChange]);

  useEffect(() => {
    if (!request) return;
    if (submitError) {
      toast.error(
        `Could not start the image build: ${getErrorMessage(submitError) || 'unknown error'}`
      );
      setRequest(null);
      return;
    }
    if (!ownJob) return;
    if (isComplete && image) {
      deploy({
        workspace,
        data: {
          agent: agentName,
          deployment_mode: request.mode,
          image,
          ...(request.deploymentName ? { name: request.deploymentName } : {}),
        },
      });
      setRequest(null);
      return;
    }
    if (isFailed || isUnreachable || resultError || (isComplete && !isResultPending && !image)) {
      toast.error(`The image build for "${agentName}" did not produce an image to deploy.`);
      setRequest(null);
    }
  }, [
    request,
    submitError,
    ownJob,
    isComplete,
    image,
    isFailed,
    isUnreachable,
    resultError,
    isResultPending,
    deploy,
    workspace,
    agentName,
    toast,
  ]);

  return null;
};
