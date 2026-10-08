// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The in-tab form sits behind its own flag, read at import time, so it gets its own file.
vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OPTIMIZATIONS_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OPTIMIZATION_FORM_ENABLED', 'true');
});

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute, getAgentOptimizeRoute } from '@studio/routes/utils';
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen } from '@studio/tests/util/render';
import { within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const OPTIMIZE_JOBS_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy`;

const renderDetail = (search = '?tab=optimizations') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace, agentName)}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('AgentDetailRoute optimization form', () => {
  it('promotes Optimize to the primary action and opens the strategy picker in the tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await screen.findByText('brevity-sweep-3');
    await user.click(await screen.findByRole('button', { name: 'Optimize' }));

    expect(
      await screen.findByRole('radio', { name: /Hyper-parameter optimization/ })
    ).not.toBeChecked();
    expect(screen.getByRole('radio', { name: /Upload a config/ })).toBeInTheDocument();
    expect(screen.queryByText('brevity-sweep-3')).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Optimize agent' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Optimize' })).toBeDisabled();
  });

  it('opens the strategy picker from the empty state', async () => {
    const user = userEvent.setup();
    server.use(
      http.get(OPTIMIZE_JOBS_URL, () =>
        HttpResponse.json({
          data: [],
          pagination: {
            page: 1,
            page_size: 20,
            current_page_size: 0,
            total_pages: 1,
            total_results: 0,
          },
        })
      )
    );
    renderDetail();

    const emptyState = await screen.findByTestId('entity-empty-state-first-use', undefined, {
      timeout: LG_SELECTOR_TIMEOUT,
    });
    await user.click(within(emptyState).getByRole('button', { name: 'Optimize' }));

    expect(
      await screen.findByRole('radio', { name: /Hyper-parameter optimization/ })
    ).toBeInTheDocument();
  });

  it('opens the strategy picker when arriving with ?action=optimize', async () => {
    renderRoute(undefined, {
      history: getAgentOptimizeRoute(workspace, agentName),
      routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
    });

    expect(
      await screen.findByRole('radio', { name: /Hyper-parameter optimization/ })
    ).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Optimize agent' })).not.toBeInTheDocument();
  });

  it('lists the planned strategies as disabled tiles', async () => {
    renderDetail('?tab=optimizations&view=strategy');

    expect(
      await screen.findByRole('radio', { name: /Hyper-parameter optimization/ })
    ).toBeEnabled();
    for (const name of [/Skill optimization/, /Model Routing \(Switchyard\)/]) {
      expect(screen.getByRole('radio', { name })).toBeDisabled();
    }
  });

  it('continues to the form from the hyper-parameter optimization strategy', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=strategy');

    await user.click(await screen.findByRole('radio', { name: /Hyper-parameter optimization/ }));

    expect(await screen.findByRole('button', { name: 'Run optimization' })).toBeInTheDocument();
  });

  it('opens the upload modal from the upload strategy, and stays on the picker when it closes', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=strategy');

    await user.click(await screen.findByRole('radio', { name: /Upload a config/ }));

    const dialog = await screen.findByRole('dialog', { name: 'Optimize agent' });
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    expect(screen.queryByRole('dialog', { name: 'Optimize agent' })).not.toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Upload a config/ })).toBeInTheDocument();
  });

  it('returns to the table from the strategy picker', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=strategy');

    await user.click(await screen.findByRole('button', { name: 'Optimizations' }));

    expect(await screen.findByText('brevity-sweep-3')).toBeInTheDocument();
    expect(
      screen.queryByRole('radio', { name: /Hyper-parameter optimization/ })
    ).not.toBeInTheDocument();
  });

  it('returns to the strategy picker from the form', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=form');

    await user.click(await screen.findByRole('button', { name: 'Back' }));

    expect(
      await screen.findByRole('radio', { name: /Hyper-parameter optimization/ })
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run optimization' })).not.toBeInTheDocument();
  });

  it('regenerates the name when the intent changes, until the user types one', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=form');

    await screen.findByDisplayValue(new RegExp(`^${agentName}-accuracy-`));

    await user.click(await screen.findByRole('radio', { name: /Brevity/ }));
    const nameField = await screen.findByDisplayValue(new RegExp(`^${agentName}-brevity-`));

    await user.clear(nameField);
    await user.type(nameField, 'my-own-name');
    await user.click(await screen.findByRole('radio', { name: /Cost/ }));

    expect(nameField).toHaveValue('my-own-name');
  });

  it('reshapes the search space when the intent changes', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=form');

    expect(await screen.findByText(/temperature 0\.0–0\.6/)).toBeInTheDocument();

    await user.click(await screen.findByRole('radio', { name: /Creativity/ }));

    expect(await screen.findByText(/temperature 0\.3–1\.5/)).toBeInTheDocument();
  });

  it('holds the run closed while the form is unanswered', async () => {
    renderDetail('?tab=optimizations&view=form');

    expect(
      await screen.findByText('Pick an evaluation to score trials against.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run optimization' })).toBeDisabled();
  });
});
