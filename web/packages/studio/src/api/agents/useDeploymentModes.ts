// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListDeploymentModes } from '@nemo/sdk/generated/agents/agent-deployments';
import {
  DeploymentModeAvailabilityMode,
  type DeploymentModeAvailabilityMode as DeploymentMode,
} from '@nemo/sdk/generated/agents/schema/DeploymentModeAvailabilityMode';
import type { DeploymentModeList } from '@nemo/sdk/generated/agents/schema/DeploymentModeList';
import type { UseQueryOptions } from '@tanstack/react-query';
import { useMemo } from 'react';

export { DeploymentModeAvailabilityMode };
export type { DeploymentMode };

// Preference order when a built image is handed to a deployment.
export const IMAGE_DEPLOYMENT_MODES: readonly DeploymentMode[] = [
  DeploymentModeAvailabilityMode.docker,
  DeploymentModeAvailabilityMode.k8s,
];

export type DeploymentModes =
  | { status: 'loading' }
  | { status: 'unknown' }
  | { status: 'ready'; enabled: DeploymentMode[] };

export const useDeploymentModes = (
  workspace: string,
  queryOptions?: Partial<UseQueryOptions<DeploymentModeList>>
): DeploymentModes => {
  const { data, isError } = useAgentsListDeploymentModes(workspace, { query: queryOptions });
  return useMemo<DeploymentModes>(() => {
    if (data) {
      return {
        status: 'ready',
        enabled: data.data.filter((mode) => mode.enabled).map((mode) => mode.mode),
      };
    }
    return isError ? { status: 'unknown' } : { status: 'loading' };
  }, [data, isError]);
};

export const enabledImageModes = (modes: DeploymentModes): readonly DeploymentMode[] =>
  modes.status === 'ready'
    ? IMAGE_DEPLOYMENT_MODES.filter((mode) => modes.enabled.includes(mode))
    : IMAGE_DEPLOYMENT_MODES;
