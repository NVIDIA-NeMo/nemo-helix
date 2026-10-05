// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const DEPLOYMENT_MODES_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployment-modes`;
export const EXECUTION_PROFILES_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/execution-profiles`;

export const deploymentModesResponse = (enabled: readonly string[]) => ({
  data: ['subprocess', 'docker', 'k8s'].map((mode) => ({
    mode,
    enabled: enabled.includes(mode),
    requires_image: mode !== 'subprocess',
  })),
});

export const executionProfilesResponse = (profiles: { profile: string; backend: string }[]) =>
  profiles.map((profile) => ({ provider: 'cpu', ...profile }));

export const agentDeploymentCapabilitiesHandlers = [
  http.get(DEPLOYMENT_MODES_URL, () =>
    HttpResponse.json(deploymentModesResponse(['subprocess', 'docker']))
  ),
  http.get(EXECUTION_PROFILES_URL, () =>
    HttpResponse.json(
      executionProfilesResponse([
        { profile: 'default', backend: 'docker' },
        { profile: 'default', backend: 'subprocess' },
      ])
    )
  ),
];
