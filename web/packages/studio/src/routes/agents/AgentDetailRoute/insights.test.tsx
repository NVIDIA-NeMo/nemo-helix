// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    value,
    onValueChange,
    ...props
  }: {
    value: { model: string } | null;
    onValueChange: (next: { model: string }) => void;
    'aria-label': string;
    disabled?: boolean;
  }) => (
    <input
      aria-label={props['aria-label']}
      disabled={props.disabled}
      value={value?.model ?? ''}
      onChange={(event) => onValueChange({ model: event.target.value })}
    />
  ),
}));

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { getCompassGetAnalysisConfigQueryKey } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { getCompassListInsightsQueryKey } from '@nemo/sdk/generated/compass/compass-insights';
import type { AnalysisRunResponse, InsightListItem } from '@nemo/sdk/generated/compass/schema';
import { LIST_POLL_MS } from '@studio/api/useLatestAnalysisRun';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  analysisConfigWriteHandlers,
  analysisRunHandlers,
  mockAnalysisConfig,
  mockAnalysisRunWithJob,
  mockInsights,
} from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { act, renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';
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
  afterEach(() => {
    server.events.removeAllListeners('request:start');
  });

  it('has no settings panel above the table, only an Insights settings button', async () => {
    renderDetail('?tab=insights');

    await checkActiveTab('Insights');
    expect(await screen.findByRole('button', { name: 'Insights settings' })).toBeInTheDocument();
    expect(screen.queryByText('Insights analysis')).not.toBeInTheDocument();
    expect(screen.queryByText('Latest run')).not.toBeInTheDocument();
  });

  it('edits the stored models and periodic analysis from Insights settings', async () => {
    const writes: { action: string; body?: unknown }[] = [];
    server.use(...analysisConfigWriteHandlers((call) => writes.push(call)));
    const user = userEvent.setup();
    renderDetail('?tab=insights');

    await user.click(await screen.findByRole('button', { name: 'Insights settings' }));
    const dialog = within(await screen.findByRole('dialog'));
    await waitFor(() =>
      expect(dialog.getByLabelText('Default model')).toHaveValue(mockAnalysisConfig.default_model)
    );
    expect(dialog.getByLabelText('Fast model')).toHaveValue(mockAnalysisConfig.fast_model);
    const periodic = dialog.getByRole('switch', { name: 'Run periodic analysis' });
    expect(periodic).toBeChecked();

    fireEvent.change(dialog.getByLabelText('Fast model'), {
      target: { value: 'default/new-fast' },
    });
    await user.click(periodic);
    await user.click(dialog.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(writes).toEqual([
      {
        action: 'enable',
        agent: 'react-agent',
        body: { default_model: mockAnalysisConfig.default_model, fast_model: 'default/new-fast' },
      },
      { action: 'disable', agent: 'react-agent' },
    ]);
    expect(await screen.findByText('Insights settings saved.')).toBeInTheDocument();
  });

  it('asks for both models when the agent has no insights settings yet', async () => {
    server.use(
      http.get(mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ detail: 'Not Found' }, { status: 404 })
      )
    );
    const user = userEvent.setup();
    renderDetail('?tab=insights');

    await user.click(await screen.findByRole('button', { name: 'Insights settings' }));
    const dialog = within(await screen.findByRole('dialog'));

    expect(
      await dialog.findByText(/Insights aren.t set up for this agent yet/)
    ).toBeInTheDocument();
    expect(dialog.getByRole('switch', { name: 'Run periodic analysis' })).not.toBeChecked();
    expect(dialog.getByRole('button', { name: 'Save' })).toBeDisabled();
  });

  it('shows the insight totals and periodic analysis status above the table', async () => {
    server.use(
      http.get(mockApiUrl(getCompassListInsightsQueryKey, ':workspace'), ({ request }) => {
        const params = new URL(request.url).searchParams;
        const status = params.get('status');
        const data = [
          ...mockInsights,
          { ...mockInsights[0], id: 'ins-resolved', status: 'resolved' as const },
        ].filter((insight) => !status || insight.status === status);
        return HttpResponse.json({
          data: data.slice(0, Number(params.get('page_size') ?? 20)),
          pagination: {
            page: 1,
            page_size: 20,
            current_page_size: data.length,
            total_pages: 1,
            total_results: data.length,
          },
        });
      })
    );
    renderDetail('?tab=insights');

    const tile = async (label: string) =>
      within(
        (await screen.findAllByTestId('stat-tile-surface')).find((element) =>
          within(element).queryByText(label, { exact: true })
        ) as HTMLElement
      );
    await waitFor(async () => expect((await tile('Insights')).getByText('3')).toBeInTheDocument());
    await waitFor(async () =>
      expect((await tile('Open insights')).getByText('2')).toBeInTheDocument()
    );
    expect(await (await tile('Periodic analysis')).findByText('Enabled')).toBeInTheDocument();
  });

  it('reports periodic analysis as off and not set up when the agent has no settings', async () => {
    server.use(
      http.get(mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ detail: 'Not Found' }, { status: 404 })
      )
    );
    renderDetail('?tab=insights');

    expect(await screen.findByText('Not set up')).toBeInTheDocument();
    expect(screen.getByText('Off')).toBeInTheDocument();
  });

  const sortButton = (header: RegExp) =>
    within(screen.getByRole('columnheader', { name: header })).getByRole('button');

  /** Clicks a header twice and checks the server is asked to sort by `field` both ways. */
  const expectServerSort = async (
    user: ReturnType<typeof userEvent.setup>,
    sorts: (string | null)[],
    header: RegExp,
    field: string
  ) => {
    await user.click(sortButton(header));
    await waitFor(() => expect(sorts.at(-1)?.replace(/^-/, '')).toBe(field));
    const first = sorts.at(-1);
    await user.click(sortButton(header));
    await waitFor(() => expect(sorts.at(-1)).toBe(first === field ? `-${field}` : field));
  };

  it('sorts the insights table by a clicked column on the server', async () => {
    const sorts: (string | null)[] = [];
    server.events.on('request:start', ({ request }) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith('/insights') && url.searchParams.get('page_size') !== '1') {
        sorts.push(url.searchParams.get('sort'));
      }
    });
    const user = userEvent.setup();
    renderDetail('?tab=insights');

    await screen.findByText(mockInsights[0].title);
    expect(sorts.at(-1)).toBe('-created_at');
    await expectServerSort(user, sorts, /^Insight/, 'title');
    await expectServerSort(user, sorts, /^Status/, 'status');
    expect(
      within(screen.getByRole('columnheader', { name: /^Traces/ })).queryByRole('button')
    ).not.toBeInTheDocument();
  });

  it('sorts the jobs table by a clicked column on the server', async () => {
    const sorts: (string | null)[] = [];
    server.use(...analysisRunHandlers(() => [mockAnalysisRunWithJob('react-agent', 'completed')]));
    server.events.on('request:start', ({ request }) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith('/analysis-runs') && url.searchParams.has('sort')) {
        sorts.push(url.searchParams.get('sort'));
      }
    });
    const user = userEvent.setup();
    renderDetail('?tab=insights&view=jobs');

    await screen.findByRole('row', { name: /analysis-run-1/ });
    await expectServerSort(user, sorts, /^Job/, 'name');
    await expectServerSort(user, sorts, /^Traces since/, 'since');
    await expectServerSort(user, sorts, /^Fast model/, 'fast_model');
    expect(
      within(screen.getByRole('columnheader', { name: /^Status/ })).queryByRole('button')
    ).not.toBeInTheDocument();
  });

  it('opens on the insights tab from a ?tab=insights link', async () => {
    renderDetail('?tab=insights');
    await checkActiveTab('Insights');
  });

  it("lists only this agent's insights, without an Agent column", async () => {
    const requestedAgents: (string | null)[] = [];
    server.use(
      http.get(mockApiUrl(getCompassListInsightsQueryKey, ':workspace'), ({ request }) => {
        const agent = new URL(request.url).searchParams.get('agent');
        requestedAgents.push(agent);
        const data = [
          ...mockInsights,
          { ...mockInsights[0], id: 'ins-other', title: 'Other agent insight', agent: 'other' },
        ].filter((insight) => !agent || insight.agent === agent);
        return HttpResponse.json({
          data,
          pagination: {
            page: 1,
            page_size: 20,
            current_page_size: data.length,
            total_pages: 1,
            total_results: data.length,
          },
        });
      })
    );
    renderDetail('?tab=insights');

    expect(await screen.findByText(mockInsights[0].title)).toBeInTheDocument();
    expect(screen.queryByText('Other agent insight')).not.toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Insight/ })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Agent' })).not.toBeInTheDocument();
    expect(requestedAgents).toContain('react-agent');
  });

  it('shows the insights empty state when the agent has none', async () => {
    server.use(
      http.get(mockApiUrl(getCompassListInsightsQueryKey, ':workspace'), () =>
        HttpResponse.json({
          data: [],
          pagination: {
            page: 1,
            page_size: 20,
            current_page_size: 0,
            total_pages: 0,
            total_results: 0,
          },
        })
      )
    );
    renderDetail('?tab=insights');

    expect(await screen.findByText('No insights yet')).toBeInTheDocument();
  });
});

describe('AgentDetailRoute insights tab jobs view', () => {
  it("lists the agent's analysis runs with their job status", async () => {
    server.use(
      ...analysisRunHandlers(() => [
        mockAnalysisRunWithJob('react-agent', 'active', { name: 'run-active' }),
        mockAnalysisRunWithJob('react-agent', 'completed', { name: 'run-done' }),
      ])
    );
    const user = userEvent.setup();
    renderDetail('?tab=insights');

    await user.click(await screen.findByRole('radio', { name: 'Jobs' }));

    const active = await screen.findByRole('row', { name: /run-active/ });
    expect(await within(active).findByText('Running')).toBeInTheDocument();
    const done = screen.getByRole('row', { name: /run-done/ });
    expect(await within(done).findByText('Completed')).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Insight' })).not.toBeInTheDocument();
  });

  it('opens on the Jobs view from a ?view=jobs link and shows its empty state', async () => {
    renderDetail('?tab=insights&view=jobs');

    expect(await screen.findByText('No insights jobs yet')).toBeInTheDocument();
  });
});

describe('AgentDetailRoute insights tab run progress', () => {
  let latest: AnalysisRunResponse | undefined;
  let findings: InsightListItem[];

  const runRow = () => screen.findByRole('row', { name: /analysis-run-1/ });
  const nextPoll = (ms = JOB_POLLING_INTERVAL_MS) => act(() => vi.advanceTimersByTimeAsync(ms));

  beforeEach(() => {
    latest = undefined;
    findings = mockInsights;
    server.use(
      ...analysisRunHandlers(() => (latest ? [latest] : [])),
      http.get(mockApiUrl(getCompassListInsightsQueryKey, ':workspace'), () =>
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

  it('follows a run from queued to running to completed and reports new findings', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'created');
    renderDetail('?tab=insights&view=jobs');

    expect(await within(await runRow()).findByText('Queued')).toBeInTheDocument();

    latest = mockAnalysisRunWithJob('react-agent', 'active');
    await nextPoll();
    expect(await within(await runRow()).findByText('Running')).toBeInTheDocument();

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
    expect(await within(await runRow()).findByText('Completed')).toBeInTheDocument();
  });

  it('reports a failed run', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'active');
    renderDetail('?tab=insights&view=jobs');

    expect(await within(await runRow()).findByText('Running')).toBeInTheDocument();

    latest = mockAnalysisRunWithJob('react-agent', 'error');
    await nextPoll();

    expect(
      await screen.findByText(
        'Analysis run "analysis-run-1" failed. Open the run\'s job to read its logs.'
      )
    ).toBeInTheDocument();
    expect(await within(await runRow()).findByText('Failed')).toBeInTheDocument();
  });

  it('picks up a run started outside the page on the next list poll', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    renderDetail('?tab=insights&view=jobs');

    expect(await screen.findByText('No insights jobs yet')).toBeInTheDocument();

    latest = mockAnalysisRunWithJob('react-agent', 'active');
    await nextPoll(LIST_POLL_MS);

    expect(await within(await runRow()).findByText('Running')).toBeInTheDocument();
  });

  it('does not toast for a run that had already finished when the page opened', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    latest = mockAnalysisRunWithJob('react-agent', 'completed');
    renderDetail('?tab=insights&view=jobs');

    expect(await within(await runRow()).findByText('Completed')).toBeInTheDocument();
    await nextPoll();
    expect(screen.queryByText(/Analysis complete/)).not.toBeInTheDocument();
  });
});
