// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { getInsightsListAnalysisRunsQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-runs';
import type { CreateAnalysisRunRequest } from '@nemo/sdk/generated/insights/schema';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  agentEthosHandlers,
  mockAnalysisConfig,
  mockAnalysisRunResponse,
} from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const renderDetail = (search = '') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

const captureRunRequests = () => {
  const requests: CreateAnalysisRunRequest[] = [];
  server.use(
    http.post<{ workspace: string }, CreateAnalysisRunRequest>(
      mockApiUrl(getInsightsListAnalysisRunsQueryKey, ':workspace'),
      async ({ params, request }) => {
        const body = await request.json();
        requests.push(body);
        return HttpResponse.json(mockAnalysisRunResponse(params.workspace, body));
      }
    )
  );
  return requests;
};

const checkActiveTab = async (tab: string) => {
  expect(await screen.findByRole('tab', { name: tab })).toHaveAttribute('aria-selected', 'true');
};

describe('AgentDetailRoute insights tab', () => {
  it('shows the stored analysis config when the tab is selected', async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole('tab', { name: 'Insights' }));
    await checkActiveTab('Insights');
    expect(await screen.findByText('Enabled')).toBeInTheDocument();
    expect(screen.getAllByText(mockAnalysisConfig.default_model)).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Edit' })).toBeInTheDocument();
  });

  it('opens on the insights tab from a ?tab=insights link', async () => {
    renderDetail('?tab=insights');
    await checkActiveTab('Insights');
  });

  it('sends the agent ETHOS.md with a run started from the tab', async () => {
    const user = userEvent.setup();
    server.use(...agentEthosHandlers('react-agent', '# React agent ethos'));
    const requests = captureRunRequests();
    renderDetail('?tab=insights');

    await user.click(await screen.findByRole('button', { name: 'Run analysis now' }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toEqual({
      agent: 'react-agent',
      default_model: mockAnalysisConfig.default_model,
      fast_model: mockAnalysisConfig.fast_model,
      ethos: '# React agent ethos',
    });
  });

  it('starts the run without ethos when the agent has no ethos fileset', async () => {
    const user = userEvent.setup();
    server.use(...agentEthosHandlers('react-agent', null));
    const requests = captureRunRequests();
    renderDetail('?tab=insights');

    await user.click(await screen.findByRole('button', { name: 'Run analysis now' }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).not.toHaveProperty('ethos');
  });
});
