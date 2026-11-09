// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
  vi.stubEnv('VITE_FF_OPTIMIZER_ENABLED', 'true');
});

import { getCompassGetAnalysisConfigQueryKey } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { getCompassGetStatusesAnalysisRunStatusQueryKey } from '@nemo/sdk/generated/compass/compass-analysis-run-statuses';
import type {
  AnalysisRunResponse,
  CreateAnalysisRunRequest,
} from '@nemo/sdk/generated/compass/schema';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  agentEthosHandlers,
  analysisConfigWriteHandlers,
  analysisRunCreateHandler,
  analysisRunHandlers,
  analysisRunStatusHandler,
  mockAnalysisConfig,
  mockAnalysisRunWithJob,
} from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { DAY_MS } from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { OptimizerRoute } from '@studio/routes/optimizer/OptimizerRoute';
import { getAgentDetailRoute, getOptimizerRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import { onlineManager } from '@tanstack/react-query';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    value,
    onValueChange,
    ...props
  }: {
    value: { model: string } | null;
    onValueChange: (next: { model: string }) => void;
    'aria-label': string;
    placeholder?: string;
    disabled?: boolean;
  }) => (
    <input
      aria-label={props['aria-label']}
      placeholder={props.placeholder}
      disabled={props.disabled}
      value={value?.model ?? ''}
      onChange={(event) => onValueChange({ model: event.target.value })}
    />
  ),
}));

const PERIODIC_CURSOR = '2026-08-14T09:00:00Z';

let runs: AnalysisRunResponse[];
let requests: CreateAnalysisRunRequest[];
let configWrites: string[];

const recordConfigWrites = ({ request }: { request: Request }) => {
  if (request.method !== 'GET' && request.url.includes('/analysis-configs')) {
    configWrites.push(`${request.method} ${request.url}`);
  }
};

beforeEach(() => {
  runs = [];
  requests = [];
  configWrites = [];
  server.use(
    ...analysisRunHandlers(() => runs),
    ...agentEthosHandlers('react-agent', '# React agent ethos'),
    analysisRunCreateHandler((body, response) => {
      requests.push(body);
      runs = [response, ...runs];
    })
  );
  server.events.on('request:start', recordConfigWrites);
});

afterEach(() => {
  server.events.removeListener('request:start', recordConfigWrites);
});

const agentDetailRoute = { path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> };

const openFromAgent = async () => {
  const user = userEvent.setup();
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}?tab=insights`,
    routes: [agentDetailRoute],
  });
  await user.click(await screen.findByRole('button', { name: 'Get Insights' }));
  const dialog = within(await screen.findByRole('dialog'));
  return { user, dialog };
};

const openFromInsightsPage = async () => {
  const user = userEvent.setup();
  renderRoute(undefined, {
    history: getOptimizerRoute(workspace1.workspace),
    routes: [{ path: ROUTES.workspace.optimizer, element: <OptimizerRoute /> }, agentDetailRoute],
  });
  await user.click(await screen.findByRole('button', { name: 'Get Insights' }));
  const dialog = within(await screen.findByRole('dialog'));
  return { user, dialog };
};

const pickOption = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>,
  combobox: string,
  name: string | RegExp
) => {
  await user.click(dialog.getByRole('combobox', { name: combobox }));
  await user.click(await screen.findByRole('option', { name }));
};

const pickPreset = (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>,
  name: string | RegExp
) => pickOption(user, dialog, 'Traces since', name);

const submitButton = (dialog: ReturnType<typeof within>) =>
  dialog.getByRole('button', { name: 'Get Insights' });

const submit = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>
) => {
  await waitFor(() => expect(submitButton(dialog)).toBeEnabled());
  await user.click(submitButton(dialog));
  await waitFor(() => expect(requests).toHaveLength(1));
  return requests[0];
};

const expectAbout = (since: string | undefined, expectedMs: number) => {
  expect(since).toBeDefined();
  expect(Math.abs(Date.parse(since as string) - expectedMs)).toBeLessThan(60_000);
};

describe('Get Insights modal from the agent page', () => {
  it('replaces Deploy with Get Insights on the insights tab', async () => {
    renderRoute(undefined, {
      history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}?tab=insights`,
      routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
    });

    expect(await screen.findByRole('button', { name: 'Get Insights' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Deploy' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run analysis now' })).not.toBeInTheDocument();
  });

  it('skips the agent picker and prefills the stored fast model', async () => {
    const { dialog } = await openFromAgent();

    expect(dialog.queryByRole('combobox', { name: 'Agent' })).not.toBeInTheDocument();
    expect(dialog.getByText('react-agent')).toBeInTheDocument();
    await waitFor(() => expect(dialog.getByLabelText('Fast model')).toBeEnabled());
    expect(dialog.getByLabelText('Fast model')).toHaveValue(mockAnalysisConfig.fast_model);
    expect(dialog.queryByLabelText('Default model')).not.toBeInTheDocument();
    expect(dialog.queryByText(/Prefilled from the stored analysis config/)).not.toBeInTheDocument();
    expect(
      dialog.queryByRole('combobox', { name: 'Scope to an evaluation' })
    ).not.toBeInTheDocument();
  });

  it('defaults to the last two weeks and sends the stored model pair', async () => {
    const { user, dialog } = await openFromAgent();

    expect(dialog.getByRole('combobox', { name: 'Traces since' })).toHaveTextContent(
      'Last two weeks'
    );
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - 14 * DAY_MS);
    expect(body).toMatchObject({
      agent: 'react-agent',
      default_model: mockAnalysisConfig.default_model,
      fast_model: mockAnalysisConfig.fast_model,
      ethos: '# React agent ethos',
    });
    expect(body).not.toHaveProperty('evaluation_id');
  });

  it('shows Start date only once Custom start date is picked', async () => {
    const { user, dialog } = await openFromAgent();

    expect(dialog.queryByRole('textbox', { name: 'Start date' })).not.toBeInTheDocument();
    await pickPreset(user, dialog, 'Custom start date');
    expect(dialog.getByRole('textbox', { name: 'Start date' })).toBeInTheDocument();
  });

  it('switches to the Jobs view and shows the queued run', async () => {
    const { user, dialog } = await openFromAgent();

    await submit(user, dialog);

    const row = await screen.findByRole('row', { name: /analysis-run-1/ });
    expect(await within(row).findByText('Queued')).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Insight' })).not.toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(await screen.findByText('Queued analysis run "analysis-run-1".')).toBeInTheDocument();
  });

  it('sends the full history for All history', async () => {
    const { user, dialog } = await openFromAgent();

    await pickPreset(user, dialog, 'All history');
    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('since');
  });

  it.each([
    ['Last 24 hours', 1],
    ['Last 7 days', 7],
    ['Last month', 30],
  ])('sends the matching lower bound for %s', async (preset, days) => {
    const { user, dialog } = await openFromAgent();

    await pickPreset(user, dialog, preset);
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - days * DAY_MS);
  });

  it('sends the start of the picked custom start date in local time', async () => {
    const { user, dialog } = await openFromAgent();

    await pickPreset(user, dialog, 'Custom start date');
    const input = dialog.getByRole('textbox', { name: 'Start date' });
    await user.clear(input);
    await user.type(input, '2026-10-01');
    await user.tab();
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date(2026, 9, 1).toISOString());
  });

  it('does not accept a custom start date in the future', async () => {
    const { user, dialog } = await openFromAgent();

    await pickPreset(user, dialog, 'Custom start date');
    const input = dialog.getByRole('textbox', { name: 'Start date' });
    await user.clear(input);
    await user.type(input, `${new Date().getFullYear() + 1}-01-01`);
    await user.tab();
    const body = await submit(user, dialog);

    expect(Date.parse(body.since as string)).toBeLessThanOrEqual(Date.now());
  });

  it("offers the periodic scheduler's cursor as a lower bound", async () => {
    server.use(analysisRunStatusHandler('react-agent', PERIODIC_CURSOR));
    const { user, dialog } = await openFromAgent();

    await pickPreset(user, dialog, /Since the last periodic analysis/);
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date(PERIODIC_CURSOR).toISOString());
  });

  it('blocks the run when a refetch loses the picked cursor', async () => {
    server.use(analysisRunStatusHandler('react-agent', PERIODIC_CURSOR));
    const { user, dialog } = await openFromAgent();
    await pickPreset(user, dialog, /Since the last periodic analysis/);

    server.use(
      http.get(
        mockApiUrl(getCompassGetStatusesAnalysisRunStatusQueryKey, ':workspace', ':agent'),
        () => HttpResponse.json({ detail: 'Unavailable' }, { status: 503 })
      )
    );
    onlineManager.setOnline(false);
    onlineManager.setOnline(true);

    expect(
      await dialog.findByText(
        'The last periodic analysis time could not be read. Pick another range.'
      )
    ).toBeInTheDocument();
    expect(submitButton(dialog)).toBeDisabled();
  });

  it('does not offer the cursor before the scheduler has run', async () => {
    const { user, dialog } = await openFromAgent();

    await user.click(dialog.getByRole('combobox', { name: 'Traces since' }));
    await screen.findByRole('option', { name: 'Last 24 hours' });
    expect(
      screen.queryByRole('option', { name: /Since the last periodic analysis/ })
    ).not.toBeInTheDocument();
  });

  it('sends an overridden fast model for this run without saving it to the config', async () => {
    const { user, dialog } = await openFromAgent();

    const input = dialog.getByLabelText('Fast model');
    await waitFor(() => expect(input).toBeEnabled());
    // The picker replaces its whole value on a pick; the mock input stands in for that.
    fireEvent.change(input, { target: { value: 'default/override-model' } });
    const body = await submit(user, dialog);

    expect(body.fast_model).toBe('default/override-model');
    expect(body.default_model).toBe(mockAnalysisConfig.default_model);
    expect(configWrites).toEqual([]);
  });

  it('sets up an agent with no insights settings, saving both models before the run', async () => {
    const writes: { action: string; body?: unknown }[] = [];
    server.use(
      http.get(mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ detail: 'Not Found' }, { status: 404 })
      ),
      ...analysisConfigWriteHandlers((call) => writes.push(call))
    );
    const { user, dialog } = await openFromAgent();

    expect(await dialog.findByText(/Pick the models Compass uses for/)).toBeInTheDocument();
    expect(submitButton(dialog)).toBeDisabled();
    fireEvent.change(dialog.getByLabelText('Default model'), {
      target: { value: 'default/setup-default' },
    });
    fireEvent.change(dialog.getByLabelText('Fast model'), {
      target: { value: 'default/setup-fast' },
    });
    const body = await submit(user, dialog);

    // Saving models goes through enable, so periodic analysis is switched back off straight after.
    expect(writes).toEqual([
      {
        action: 'enable',
        agent: 'react-agent',
        body: { default_model: 'default/setup-default', fast_model: 'default/setup-fast' },
      },
      { action: 'disable', agent: 'react-agent' },
    ]);
    expect(body).toMatchObject({
      default_model: 'default/setup-default',
      fast_model: 'default/setup-fast',
    });
  });

  it('asks for the default model when the stored settings have none', async () => {
    server.use(
      http.get(mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ ...mockAnalysisConfig, default_model: undefined })
      )
    );
    const { dialog } = await openFromAgent();

    expect(await dialog.findByLabelText('Default model')).toHaveValue('');
    expect(submitButton(dialog)).toBeDisabled();
  });

  it('does not run when saving the setup fails', async () => {
    server.use(
      http.get(mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ detail: 'Not Found' }, { status: 404 })
      ),
      http.post(
        `${mockApiUrl(getCompassGetAnalysisConfigQueryKey, ':workspace', ':agent')}/enable`,
        () => HttpResponse.json({ detail: 'Boom' }, { status: 500 })
      )
    );
    const { user, dialog } = await openFromAgent();

    await dialog.findByText(/Pick the models Compass uses for/);
    fireEvent.change(dialog.getByLabelText('Default model'), {
      target: { value: 'default/setup-default' },
    });
    fireEvent.change(dialog.getByLabelText('Fast model'), {
      target: { value: 'default/setup-fast' },
    });
    await waitFor(() => expect(submitButton(dialog)).toBeEnabled());
    await user.click(submitButton(dialog));

    expect(await screen.findByRole('dialog')).toBeInTheDocument();
    await waitFor(() => expect(submitButton(dialog)).toBeEnabled());
    expect(requests).toHaveLength(0);
  });

  it('blocks a new run while one is in progress', async () => {
    runs = [mockAnalysisRunWithJob('react-agent', 'active')];
    const { dialog } = await openFromAgent();

    expect(await dialog.findByText(/is still in progress/)).toBeInTheDocument();
    expect(submitButton(dialog)).toBeDisabled();
  });

  it('omits the Ethos when the agent has no ETHOS.md', async () => {
    server.use(...agentEthosHandlers('react-agent', null));
    const { user, dialog } = await openFromAgent();

    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('ethos');
  });
});

describe('Get Insights modal from the Insights page', () => {
  it('requires an agent before it can run', async () => {
    const { dialog } = await openFromInsightsPage();

    expect(dialog.getByRole('combobox', { name: 'Agent' })).toBeEnabled();
    expect(submitButton(dialog)).toBeDisabled();
  });

  it('asks for agents within the API page size limit', async () => {
    const pageSizes: number[] = [];
    const recordAgentList = ({ request }: { request: Request }) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith('/agents'))
        pageSizes.push(Number(url.searchParams.get('page_size')));
    };
    server.events.on('request:start', recordAgentList);
    try {
      await openFromInsightsPage();
      await waitFor(() => expect(pageSizes.length).toBeGreaterThan(0));
      // The agents API rejects page_size above 100 with a 422.
      expect(Math.max(...pageSizes)).toBeLessThanOrEqual(100);
    } finally {
      server.events.removeListener('request:start', recordAgentList);
    }
  });

  it('lists the registered agents and runs for the one picked', async () => {
    const { user, dialog } = await openFromInsightsPage();

    await user.click(dialog.getByRole('combobox', { name: 'Agent' }));
    expect(await screen.findByRole('option', { name: 'react-agent2' })).toBeInTheDocument();
    await user.click(screen.getByRole('option', { name: 'react-agent' }));
    const body = await submit(user, dialog);

    expect(body.agent).toBe('react-agent');
    expect(await screen.findByRole('tab', { name: 'Insights' })).toHaveAttribute(
      'aria-selected',
      'true'
    );
    expect(await screen.findByRole('row', { name: /analysis-run-1/ })).toBeInTheDocument();
  });
});
