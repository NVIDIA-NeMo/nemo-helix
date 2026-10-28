// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const AGENT_DEPLOYMENTS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/deployments`;

type MockAgentDeployment = Pick<
  AgentDeployment,
  'name' | 'workspace' | 'agent' | 'status' | 'deployment_mode'
>;

export const agentDeployment = (
  agent: string,
  name: string,
  overrides: Partial<Pick<AgentDeployment, 'status' | 'deployment_mode'>> = {}
): MockAgentDeployment => ({
  name,
  workspace: 'default',
  agent,
  status: 'running',
  deployment_mode: 'subprocess',
  ...overrides,
});

export const agentDeploymentsHandler = (deployments: MockAgentDeployment[]) =>
  http.get(AGENT_DEPLOYMENTS_URL, () => HttpResponse.json({ data: deployments }));
