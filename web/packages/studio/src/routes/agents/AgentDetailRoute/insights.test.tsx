// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { getInsightsListInsightsQueryKey } from '@nemo/sdk/generated/insights/insights-insights';
import type { AnalysisRunResponse, InsightListItem } from '@nemo/sdk/generated/insights/schema';
import { LIST_POLL_MS } from '@studio/api/useLatestAnalysisRun';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  analysisRunCreateHandler,
  analysisRunHandlers,
  mockAnalysisConfig,
  mockAnalysisRunWithJob,
  mockInsights,
} from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { act, renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const renderDetail = (search = '') =>
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}${search}`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });

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
});

describe('AgentDetailRoute insights tab latest analysis run', () => {
  let latest: AnalysisRunResponse | undefined;
  let findings: InsightListItem[];

  const runButton = () => screen.findByRole('button', { name: 'Run analysis now' });
  const nextPoll = (ms = JOB_POLLING_INTERVAL_MS) => act(() => vi.advanceTimersByTimeAsync(ms));

  beforeEach(() => {
    latest = undefined;
    findings = mockInsights;
    server.use(
      ...analysisRunHandlers(() => (latest ? [latest] : [])),
      http.get(mockApiUrl(getInsightsListInsightsQueryKey, ':workspace'), () =>
        HttpResponse.json({
          data: findings,
          pagination: {
            page: 1,
            page_size: 100,
            current_page_size: findings.length,
            total_pages: 1,
            total_results: findings.length,
          },
        })
      )
    );
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows the queued run and disables Run analysis now once a run is started', async () => {
    const user = userEvent.setup();
    server.use(
      analysisRunCreateHandler((_body, response) => {
        latest = response;
      })
    );
    renderDetail('?tab=insights');

    await user.click(await runButton());
    await user.click(await screen.findByRole('button', { name: 'Run insight analysis' }));

    expect(await screen.findByText('Queued')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(await screen.findByText('Queued analysis run "analysis-run-1".')).toBeInTheDocument();
    await waitFor(async () => expect(await runButton()).toBeDisabled());
    expect(screen.getByText(/You can start another run when this one finishes/)).toBeVisible();
  });

  it('follows a run from queued to running to completed and reports new findings', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'created');
    renderDetail('?tab=insights');

    expect(await screen.findByText('Queued')).toBeInTheDocument();
    expect(await runButton()).toBeDisabled();

    latest = mockAnalysisRunWithJob('react-agent', 'active');
    await nextPoll();
    expect(await screen.findByText('Running')).toBeInTheDocument();
    expect(await runButton()).toBeDisabled();

    findings = [
      {
        ...mockInsights[0],
        id: 'ins-new',
        name: 'new-finding',
        created_at: '2026-08-14T09:30:00Z',
      },
      ...mockInsights,
    ];
    latest = mockAnalysisRunWithJob('react-agent', 'completed');
    await nextPoll();

    expect(await screen.findByText('Analysis complete: 1 new finding.')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(await runButton()).toBeEnabled();
    expect(screen.getByRole('button', { name: 'View job' })).toBeInTheDocument();
  });

  it('reports a failed run and re-enables Run analysis now', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'active');
    renderDetail('?tab=insights');

    expect(await screen.findByText('Running')).toBeInTheDocument();

    latest = mockAnalysisRunWithJob('react-agent', 'error');
    await nextPoll();

    expect(
      await screen.findByText(
        'Analysis run "analysis-run-1" failed. Open the run\'s job to read its logs.'
      )
    ).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
    expect(await runButton()).toBeEnabled();
  });

  it('picks up a run started outside the page on the next list poll', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    renderDetail('?tab=insights');

    expect(await screen.findByText('Enabled')).toBeInTheDocument();
    await waitFor(async () => expect(await runButton()).toBeEnabled());

    latest = mockAnalysisRunWithJob('react-agent', 'active');
    await nextPoll(LIST_POLL_MS);

    expect(await screen.findByText('Running')).toBeInTheDocument();
    expect(await runButton()).toBeDisabled();
  });

  it('does not block new runs on a paused job', async () => {
    latest = mockAnalysisRunWithJob('react-agent', 'paused');
    renderDetail('?tab=insights');

    expect(await screen.findByTestId('latest-analysis-run')).toBeInTheDocument();
    await waitFor(async () => expect(await runButton()).toBeEnabled());
    expect(screen.queryByText(/You can start another run/)).not.toBeInTheDocument();
  });

  it('does not toast for a run that had already finished when the page opened', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'completed');
    renderDetail('?tab=insights');

    expect(await screen.findByText('Completed')).toBeInTheDocument();
    await nextPoll();
    expect(screen.queryByText(/Analysis complete/)).not.toBeInTheDocument();
  });
});
