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
  it('promotes Optimize to the primary action and opens the form in the tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await screen.findByText('brevity-sweep-3');
    await user.click(await screen.findByRole('button', { name: 'Optimize' }));

    expect(await screen.findByText('New optimization')).toBeInTheDocument();
    expect(screen.queryByText('brevity-sweep-3')).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Optimize agent' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Optimize' })).toBeDisabled();
  });

  it('opens the form from the empty state', async () => {
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

    expect(await screen.findByText('New optimization')).toBeInTheDocument();
  });

  it('opens the form when arriving with ?action=optimize', async () => {
    renderRoute(undefined, {
      history: getAgentOptimizeRoute(workspace, agentName),
      routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
    });

    expect(await screen.findByText('New optimization')).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Optimize agent' })).not.toBeInTheDocument();
  });

  it('returns to the table from the form breadcrumb', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=new');

    await user.click(await screen.findByRole('button', { name: 'Optimizations' }));

    expect(await screen.findByText('brevity-sweep-3')).toBeInTheDocument();
    expect(screen.queryByText('New optimization')).not.toBeInTheDocument();
  });

  it('regenerates the name when the intent changes, until the user types one', async () => {
    const user = userEvent.setup();
    renderDetail('?tab=optimizations&view=new');

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
    renderDetail('?tab=optimizations&view=new');

    // Accuracy is the default intent.
    expect(await screen.findByText(/temperature 0\.0–0\.6 · 1 parameter/)).toBeInTheDocument();

    await user.click(await screen.findByRole('radio', { name: /Creativity/ }));

    expect(await screen.findByText(/temperature 0\.3–1\.5 · 1 parameter/)).toBeInTheDocument();
  });

  it('holds the run closed while the form is unanswered', async () => {
    renderDetail('?tab=optimizations&view=new');

    // This agent has no published evaluations, so picking one is the first unanswered question.
    expect(
      await screen.findByText('Pick an evaluation to score trials against.')
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run optimization' })).toBeDisabled();
  });
});
