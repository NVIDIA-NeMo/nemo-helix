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
import { LG_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';

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
    await waitFor(
      () => {
        expect(screen.getByText('brevity-sweep-3')).toBeInTheDocument();
        expect(screen.getByText('accuracy-sweep-1')).toBeInTheDocument();
        expect(screen.queryByText('other-agent-sweep')).not.toBeInTheDocument();
      },
      { timeout: LG_SELECTOR_TIMEOUT }
    );
  });

  it('opens the launch modal from the Optimize button', async () => {
    renderDetail();

    const optimize = await screen.findByRole('button', { name: 'Optimize' });
    await waitFor(() => expect(optimize).toBeEnabled());
    fireEvent.click(optimize);

    expect(await screen.findByRole('dialog', { name: 'Optimize agent' })).toBeInTheDocument();
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
      await screen.findByText('brevity-sweep-3', undefined, { timeout: LG_SELECTOR_TIMEOUT });

      await waitFor(() =>
        expect(filters).toContain(
          JSON.stringify({ 'spec.agent': { $in: [agentName, `${workspace}/${agentName}`] } })
        )
      );
    } finally {
      server.events.removeListener('request:start', capture);
    }
  });
});
