// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OPTIMIZATIONS_ENABLED', 'true');
});

import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

const agentName = 'react-agent';
const workspace = workspace1.workspace;

const renderDetail = (search = '?tab=optimizations') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace, agentName)}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('AgentDetailRoute optimizations tab', () => {
  it('lists only this agent’s studies', async () => {
    renderDetail();

    expect(await screen.findByRole('tab', { name: 'Optimizations' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(await screen.findByText('brevity-sweep-3')).toBeInTheDocument();
    expect(screen.getByText('accuracy-sweep-1')).toBeInTheDocument();
    expect(screen.queryByText('other-agent-sweep')).not.toBeInTheDocument();
  });

  it('scopes the list server-side with a spec.agent filter', async () => {
    const filters: string[] = [];
    const capture = ({ request }: { request: Request }) => {
      const url = new URL(request.url);
      if (!url.pathname.endsWith('/jobs/optimize')) return;
      filters.push(url.searchParams.get('filter') ?? '');
    };
    server.events.on('request:start', capture);

    try {
      renderDetail();
      await screen.findByText('brevity-sweep-3');

      await waitFor(() =>
        expect(filters).toContain(
          JSON.stringify({ 'spec.agent': { $in: [agentName, `${workspace}/${agentName}`] } })
        )
      );
    } finally {
      server.events.removeListener('request:start', capture);
    }
  });

  it('promotes Optimize to the primary action and opens the form in the tab', async () => {
    const user = userEvent.setup();
    renderDetail();

    await screen.findByText('brevity-sweep-3');
    await user.click(await screen.findByRole('button', { name: 'Optimize' }));

    expect(await screen.findByText('New optimization')).toBeInTheDocument();
    // The form replaces the table rather than layering over it.
    expect(screen.queryByText('brevity-sweep-3')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Optimize' })).not.toBeInTheDocument();
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

    expect(await screen.findByText('Pick a judge model to score trials with.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run optimization' })).toBeDisabled();
  });
});
