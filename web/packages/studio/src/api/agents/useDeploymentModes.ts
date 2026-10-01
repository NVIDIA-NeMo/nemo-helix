// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListDeploymentModes } from '@nemo/sdk/generated/agents/agent-deployments';
import {
  DeploymentModeAvailabilityMode,
  type DeploymentModeAvailabilityMode as DeploymentMode,
} from '@nemo/sdk/generated/agents/schema/DeploymentModeAvailabilityMode';
import { useMemo } from 'react';

export type { DeploymentMode };

// Preference order when a built image is handed to a deployment.
export const IMAGE_DEPLOYMENT_MODES: readonly DeploymentMode[] = [
  DeploymentModeAvailabilityMode.docker,
  DeploymentModeAvailabilityMode.k8s,
];

export const DEPLOYMENT_MODE_LABELS: Record<DeploymentMode, string> = {
  subprocess: 'Subprocess',
  docker: 'Docker',
  k8s: 'Kubernetes',
};

export type DeploymentModes =
  | { status: 'loading' }
  | { status: 'unknown' }
  | { status: 'ready'; enabled: DeploymentMode[] };

export const useDeploymentModes = (
  workspace: string,
  { enabled = true }: { enabled?: boolean } = {}
): DeploymentModes => {
  const { data, isError } = useAgentsListDeploymentModes(workspace, { query: { enabled } });
  return useMemo<DeploymentModes>(() => {
    if (isError) return { status: 'unknown' };
    if (!data) return { status: 'loading' };
    return {
      status: 'ready',
      enabled: data.data.filter((mode) => mode.enabled).map((mode) => mode.mode),
    };
  }, [data, isError]);
};

export const enabledImageModes = (modes: DeploymentModes): readonly DeploymentMode[] =>
  modes.status === 'ready'
    ? IMAGE_DEPLOYMENT_MODES.filter((mode) => modes.enabled.includes(mode))
    : IMAGE_DEPLOYMENT_MODES;
