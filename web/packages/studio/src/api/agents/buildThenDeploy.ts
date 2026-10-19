// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type DeploymentMode,
  DeploymentModeAvailabilityMode,
} from '@studio/api/agents/useDeploymentModes';
import { getAgentDetailRoute } from '@studio/routes/utils';

export interface BuildThenDeployRequest {
  mode: DeploymentMode;
  deploymentName?: string;
}

interface BuildThenDeployState {
  buildThenDeploy: BuildThenDeployRequest;
}

const isDeploymentMode = (value: unknown): value is DeploymentMode =>
  Object.values<unknown>(DeploymentModeAvailabilityMode).includes(value);

export const getBuildThenDeployRequest = (state: unknown): BuildThenDeployRequest | null => {
  const request = (state as Partial<BuildThenDeployState> | null)?.buildThenDeploy as unknown;
  if (!request || typeof request !== 'object') return null;
  const { mode, deploymentName } = request as Record<string, unknown>;
  if (!isDeploymentMode(mode)) return null;
  return typeof deploymentName === 'string' && deploymentName ? { mode, deploymentName } : { mode };
};

/** Where to navigate, and with what state, so the agent's page builds an image and deploys it. */
export const buildThenDeployNavigation = (
  workspace: string,
  agentName: string,
  request: BuildThenDeployRequest
): [string, { state: BuildThenDeployState }] => [
  `${getAgentDetailRoute(workspace, agentName)}?tab=deployments`,
  { state: { buildThenDeploy: request } },
];
