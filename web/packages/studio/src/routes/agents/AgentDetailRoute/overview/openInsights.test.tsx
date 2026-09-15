// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const agentName = 'react-agent';
const workspace = workspace1.workspace;
const INSIGHTS_URL = mockApiUrl(getInsightsListInsightsQueryKey, ':workspace');

const renderDetail = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [
      { path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> },
      { path: ROUTES.workspace.optimizer, element: <div>insights-list-page</div> },
      { path: ROUTES.workspace.optimizerInsight, element: <div>insight-detail-page</div> },
    ],
  });

describe('Open insights on the agent overview', () => {
  it('lists the agent open insights with their evidence', async () => {
    renderDetail();

    expect(await screen.findByText('Open insights')).toBeInTheDocument();
    expect(
      await screen.findByText('Ambiguous system prompt causes tool misfires')
    ).toBeInTheDocument();
    expect(screen.getByText('Latency spikes on long context (>8k tokens)')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
    expect(screen.getAllByText('Traces')).toHaveLength(2);
    expect(screen.getByText('2 total')).toBeInTheDocument();
  });

  it('ranks insights by evidence volume', async () => {
    renderDetail();

    const titles = await screen.findAllByText(/Ambiguous system prompt|Latency spikes/);
    expect(titles[0]).toHaveTextContent('Ambiguous system prompt causes tool misfires');
  });

  it('scopes the request to the agent and to open insights', async () => {
    const request = vi.fn();
    server.use(
      http.get(INSIGHTS_URL, ({ request: req }) => {
        request(Object.fromEntries(new URL(req.url).searchParams));
        return HttpResponse.json({ data: [], pagination: { total_results: 0 } });
      })
    );

    renderDetail();

    await screen.findByText(/Insights are filed by the analyst/);
    expect(request).toHaveBeenCalledWith(
      expect.objectContaining({ agent: agentName, status: 'open' })
    );
  });

  it('opens an insight detail page from a row', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(
      await screen.findByRole('button', { name: /Ambiguous system prompt causes tool misfires/ })
    );

    expect(await screen.findByText('insight-detail-page')).toBeInTheDocument();
  });

  it('navigates to the full insights list from View all', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('button', { name: 'View all' }));

    expect(await screen.findByText('insights-list-page')).toBeInTheDocument();
  });

  it('shows an empty state when the agent has no open insights', async () => {
    server.use(
      http.get(INSIGHTS_URL, () =>
        HttpResponse.json({ data: [], pagination: { total_results: 0 } })
      )
    );

    renderDetail();

    expect(await screen.findByText(/Insights are filed by the analyst/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View all' })).not.toBeInTheDocument();
  });

  it('scopes a failing insights service to Insights, not to trace statistics', async () => {
    server.use(http.get(INSIGHTS_URL, () => HttpResponse.json({}, { status: 500 })));

    renderDetail();

    expect(await screen.findByText("Insights couldn't be loaded right now.")).toBeInTheDocument();
    expect(screen.getByText('Open insights')).toBeInTheDocument();
    expect(screen.queryByText('Trace statistics are unavailable')).not.toBeInTheDocument();
  });
});
