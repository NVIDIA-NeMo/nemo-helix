// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { AgentsListRoute } from '@studio/routes/agents/AgentsListRoute';
import { getAgentsListRoute } from '@studio/routes/utils';
import { renderRoute } from '@studio/tests/util/render';
import { fireEvent, screen } from '@testing-library/react';
import { useLocation } from 'react-router';

vi.mock('@studio/plugins/PluginContext', () => ({
  usePluginInstalled: () => true,
  usePluginsError: () => null,
  usePluginsLoaded: () => true,
}));

vi.mock('@studio/components/dataViews/AgentsDataView', () => ({
  AgentsTable: () => <div>Agents table</div>,
}));

vi.mock('@studio/routes/agents/AgentsListRoute/NewAgentModal', () => ({
  NewAgentModal: ({ open }: { open: boolean }) =>
    open ? <div role="dialog">Register an agent</div> : null,
}));

vi.mock('@studio/routes/agents/AgentsListRoute/CloneAgentModal', () => ({
  CloneAgentModal: () => null,
}));

vi.mock('@studio/routes/agents/AgentDeploymentsListRoute/CreateDeploymentModal', () => ({
  CreateDeploymentModal: () => null,
}));

const workspace = 'my-workspace';

const LocationProbe = () => {
  const location = useLocation();
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>;
};

const renderAgentsList = (history: string) =>
  renderRoute(undefined, {
    history,
    routes: [
      {
        path: ROUTES.workspace.agentsList,
        element: (
          <>
            <AgentsListRoute />
            <LocationProbe />
          </>
        ),
      },
    ],
  });

describe('AgentsListRoute', () => {
  it('keeps the register dialog closed by default', async () => {
    renderAgentsList(getAgentsListRoute(workspace));

    expect(await screen.findByText('Agents table')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('opens the register dialog from the Register Agent button', async () => {
    renderAgentsList(getAgentsListRoute(workspace));

    fireEvent.click(await screen.findByRole('button', { name: 'Register Agent' }));

    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('opens the register dialog when deep-linked and strips the param from the URL', async () => {
    renderAgentsList(getAgentsListRoute(workspace, { register: true }));

    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent(
      /^\/workspaces\/my-workspace\/agents$/
    );
  });
});
