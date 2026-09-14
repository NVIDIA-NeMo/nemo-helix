// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_INTAKE_ENABLED', 'true');
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import { getGetTraceMetricsQueryKey } from '@nemo/sdk/generated/platform/traces';
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
const METRICS_URL = mockApiUrl(getGetTraceMetricsQueryKey, ':workspace');
const INSIGHTS_URL = mockApiUrl(getInsightsListInsightsQueryKey, ':workspace');

/** An agent that has never reported: no trace rollups, and therefore no insights. */
const withoutTelemetry = () => {
  server.use(
    http.get(METRICS_URL, ({ request }) => {
      const bucket = new URL(request.url).searchParams.get('bucket') ?? 'total';
      return HttpResponse.json({ bucket, timezone: 'UTC', data: [] });
    }),
    http.get(INSIGHTS_URL, () => HttpResponse.json({ data: [], pagination: { total_results: 0 } }))
  );
};

const renderDetail = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

describe('Agent overview before the agent reports anything', () => {
  it('replaces trace statistics with the two ways to connect the agent', async () => {
    withoutTelemetry();
    renderDetail();

    expect(await screen.findByText('Get started with agent optimization')).toBeInTheDocument();
    expect(screen.getByText('Begin with traces')).toBeInTheDocument();
    expect(screen.getByText('Integrate your agent')).toBeInTheDocument();
    expect(screen.queryByText('Trace statistics')).not.toBeInTheDocument();
  });

  it('copies a coding agent prompt scoped to this agent and workspace', async () => {
    // `userEvent.setup()` stubs `navigator.clipboard`, so the prompt is readable back from it.
    const user = userEvent.setup();
    withoutTelemetry();
    renderDetail();

    await user.click(
      await screen.findByRole('button', { name: 'Get coding agent prompt for importing traces' })
    );
    // The prompt editor is a lazily imported chunk, so it can miss the default find timeout.
    await user.click(
      await screen.findByRole('button', { name: 'Copy to clipboard' }, { timeout: 10_000 })
    );

    const prompt = await navigator.clipboard.readText();
    expect(prompt).toContain('nemo-intake');
    expect(prompt).toContain(agentName);
    expect(prompt).toContain(workspace);
  });

  it('points the insights panel at connecting the agent rather than at the analyst', async () => {
    withoutTelemetry();
    renderDetail();

    expect(
      await screen.findByText(
        'Generating insights requires importing traces or integrating your agent with NeMo Platform.'
      )
    ).toBeInTheDocument();
    expect(screen.getByText('0 total')).toBeInTheDocument();
    expect(screen.queryByText('No open insights')).not.toBeInTheDocument();
  });

  it('shows trace statistics once the agent has reported', async () => {
    renderDetail();

    expect(await screen.findByText('Trace statistics')).toBeInTheDocument();
    expect(screen.queryByText('Get started with agent optimization')).not.toBeInTheDocument();
  });
});
