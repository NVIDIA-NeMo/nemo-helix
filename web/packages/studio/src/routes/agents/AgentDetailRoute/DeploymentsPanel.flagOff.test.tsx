// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_CONTAINER_DEPLOYMENTS_ENABLED', 'false');
});

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { DeploymentsPanel } from '@studio/routes/agents/AgentDetailRoute/DeploymentsPanel';
import { renderRoute, screen } from '@studio/tests/util/render';

describe('DeploymentsPanel without container deployments', () => {
  it('keeps packaging out of the way, since there is nothing to deploy an image with', () => {
    renderRoute(
      <DeploymentsPanel
        agentName="calculator-agent"
        deployments={[{ name: 'calc-dep', status: 'running' } as AgentDeployment]}
        isDeploymentsLoading={false}
        isDeploying={false}
        onDeploy={vi.fn()}
        onChat={vi.fn()}
        onDelete={vi.fn()}
        onViewLogs={vi.fn()}
        canDeploy
        workspace="default"
        canPackage
      />
    );

    expect(
      screen.queryByRole('button', { name: /Build image|Manage image/ })
    ).not.toBeInTheDocument();
  });
});
