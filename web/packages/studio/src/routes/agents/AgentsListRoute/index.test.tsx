// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { PluginContext } from '@studio/plugins/PluginContext';
import { AgentsListRoute } from '@studio/routes/agents/AgentsListRoute';
import { getAgentsListRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const workspace = workspace1.workspace;

const renderAgentsList = (history: string) =>
  renderRoute(undefined, {
    history,
    routes: [
      {
        path: ROUTES.workspace.agentsList,
        element: (
          <PluginContext.Provider
            value={{
              plugins: [],
              installedNames: new Set(['agents']),
              isLoaded: true,
              isError: false,
            }}
          >
            <AgentsListRoute />
            <LocationDisplay />
          </PluginContext.Provider>
        ),
      },
    ],
  });

describe('AgentsListRoute register deep link', () => {
  it('opens the register modal on the requested tab, then drops the param', async () => {
    renderAgentsList(getAgentsListRoute(workspace, { register: 'github' }));

    const dialog = await screen.findByRole('dialog', { name: /Register an agent/ });

    expect(
      within(dialog).getByRole('tab', { name: 'Register from GitHub', selected: true })
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
        new RegExp(`^${getAgentsListRoute(workspace)}$`)
      )
    );
  });

  it('opens Register Agent on the upload tab after a deep-linked open is closed', async () => {
    const user = userEvent.setup();
    renderAgentsList(getAgentsListRoute(workspace, { register: 'github' }));

    const dialog = await screen.findByRole('dialog', { name: /Register an agent/ });
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: /Register an agent/ })).not.toBeInTheDocument()
    );

    await user.click(screen.getByRole('button', { name: 'Register Agent' }));

    const reopened = await screen.findByRole('dialog', { name: /Register an agent/ });
    expect(
      within(reopened).getByRole('tab', { name: 'Register with code upload', selected: true })
    ).toBeInTheDocument();
  });

  it('ignores a tab it does not know', async () => {
    renderAgentsList(`${getAgentsListRoute(workspace)}?register=nonsense`);

    expect(await screen.findByRole('button', { name: 'Register Agent' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: /Register an agent/ })).not.toBeInTheDocument();
  });
});
