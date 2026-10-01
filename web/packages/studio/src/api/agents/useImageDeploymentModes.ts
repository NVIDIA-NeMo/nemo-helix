// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListDeploymentModes } from '@nemo/sdk/generated/agents/agent-deployments';
import {
  DeploymentModeAvailabilityMode,
  type DeploymentModeAvailabilityMode as DeploymentMode,
} from '@nemo/sdk/generated/agents/schema/DeploymentModeAvailabilityMode';

const IMAGE_DEPLOYMENT_MODES: readonly DeploymentMode[] = [
  DeploymentModeAvailabilityMode.k8s,
  DeploymentModeAvailabilityMode.docker,
];

// Undefined while loading or unreadable, so callers keep offering image builds.
export const useImageDeploymentModes = (workspace: string): DeploymentMode[] | undefined => {
  const { data } = useAgentsListDeploymentModes(workspace);
  if (!data) return undefined;
  return IMAGE_DEPLOYMENT_MODES.filter((mode) =>
    data.data.some((availability) => availability.mode === mode && availability.enabled)
  );
};
