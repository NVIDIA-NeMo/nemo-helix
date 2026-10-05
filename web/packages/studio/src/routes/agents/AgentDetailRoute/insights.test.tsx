// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockAnalysisConfig } from '@studio/mocks/handlers/insights';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const renderDetail = (search = '') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('AgentDetailRoute insights tab', () => {
  it('shows the stored analysis config when the tab is selected', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Insights' }));

    expect(await screen.findByText('Insights analysis')).toBeInTheDocument();
    expect(await screen.findByText('Enabled')).toBeInTheDocument();
    expect(screen.getAllByText(mockAnalysisConfig.default_model)).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
  });

  it('opens on the insights tab from a ?tab=insights link', async () => {
    renderDetail('?tab=insights');

    expect(await screen.findByRole('tab', { name: 'Insights' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(await screen.findByText('Insights analysis')).toBeInTheDocument();
  });
});
