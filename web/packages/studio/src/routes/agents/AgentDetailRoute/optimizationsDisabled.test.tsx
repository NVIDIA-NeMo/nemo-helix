// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OPTIMIZATIONS_ENABLED', 'false');
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

describe('AgentDetailRoute with the optimizations flag off', () => {
  it('hides the optimizations tab', async () => {
    renderDetail();

    expect(await screen.findByTestId('nv-page-header-heading')).toHaveTextContent(agentName);
    expect(screen.queryByRole('tab', { name: 'Optimizations' })).not.toBeInTheDocument();
  });

  it('falls back to the default tab for a stale ?tab=optimizations link', async () => {
    renderDetail('?tab=optimizations');

    expect(await screen.findByRole('tab', { name: 'Deployments' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
  });
});
