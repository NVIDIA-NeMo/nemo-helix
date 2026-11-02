// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'false');
});

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const renderDetail = (search = '') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace, agentName)}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('AgentDetailRoute with the overview flag off', () => {
  it('hides the overview tab and lands on deployments', async () => {
    renderDetail();

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent(agentName);
    expect(screen.queryByRole('tab', { name: 'Overview' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Deployment' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });

  it('falls back to deployments for a stale ?tab=overview link', async () => {
    renderDetail('?tab=overview');

    expect(await screen.findByRole('tab', { name: 'Deployment' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });
});
