// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
});

import { getInsightsGetAnalysisConfigQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-configs';
import { getInsightsGetStatusesAnalysisRunStatusQueryKey } from '@nemo/sdk/generated/insights/insights-analysis-run-statuses';
import type {
  AnalysisRunResponse,
  CreateAnalysisRunRequest,
} from '@nemo/sdk/generated/insights/schema';
import { queryClientConfig } from '@studio/api/queryClient';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  agentEthosHandlers,
  agentEvaluationsHandler,
  analysisRunCreateHandler,
  analysisRunHandlers,
  analysisRunStatusHandler,
  mockAnalysisConfig,
} from '@studio/mocks/handlers/insights';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { DAY_MS } from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import { onlineManager } from '@tanstack/react-query';
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
  }) => (
    <input
      aria-label={props['aria-label']}
      placeholder={props.placeholder}
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
    analysisRunCreateHandler((body) => {
      requests.push(body);
    })
  );
  server.events.on('request:start', recordConfigWrites);
});

afterEach(() => {
  server.events.removeListener('request:start', recordConfigWrites);
});

const openModal = async ({ productionCache = false } = {}) => {
  const user = userEvent.setup();
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}?tab=insights`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
    ...(productionCache ? { testProviderOptions: { queryClientConfig } } : {}),
  });
  await user.click(await screen.findByRole('button', { name: 'Run analysis now' }));
  const dialog = within(await screen.findByRole('dialog'));
  return { user, dialog };
};

const pickPreset = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>,
  name: string | RegExp
) => {
  await user.click(dialog.getByRole('combobox', { name: 'Traces to analyze' }));
  await user.click(await screen.findByRole('option', { name }));
};

const submit = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>
) => {
  const button = dialog.getByRole('button', { name: 'Run insight analysis' });
  await waitFor(() => expect(button).toBeEnabled());
  await user.click(button);
  await waitFor(() => expect(requests).toHaveLength(1));
  return requests[0];
};

const expectAbout = (since: string | undefined, expectedMs: number) => {
  expect(since).toBeDefined();
  expect(Math.abs(Date.parse(since as string) - expectedMs)).toBeLessThan(60_000);
};

describe('Run analysis modal', () => {
  it('opens with blank model pickers that name the stored models and no Ethos control', async () => {
    const { dialog } = await openModal();

    await waitFor(() =>
      expect(dialog.getByRole('button', { name: 'Run insight analysis' })).toBeEnabled()
    );
    expect(dialog.getByLabelText('Default model')).toHaveValue('');
    expect(dialog.getByLabelText('Default model')).toHaveAttribute(
      'placeholder',
      'Default (nvidia-nemotron-mini-4b-instruct)'
    );
    expect(dialog.getByLabelText('Fast model')).toHaveValue('');
    expect(dialog.queryByRole('button', { name: 'Use default' })).not.toBeInTheDocument();
    expect(dialog.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(dialog.queryByText(/ETHOS\.md/)).not.toBeInTheDocument();
  });

  it('defaults to the full trace history', async () => {
    const { user, dialog } = await openModal();

    expect(dialog.getByRole('combobox', { name: 'Traces to analyze' })).toHaveTextContent(
      'All history'
    );
    expect(dialog.getByText("Analyzes the agent's full trace history.")).toBeInTheDocument();
    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('since');
    expect(body).toMatchObject({
      agent: 'react-agent',
      default_model: mockAnalysisConfig.default_model,
      fast_model: mockAnalysisConfig.fast_model,
      ethos: '# React agent ethos',
    });
    expect(body).not.toHaveProperty('evaluation_id');
  });

  it("offers the periodic scheduler's cursor as a lower bound", async () => {
    server.use(analysisRunStatusHandler('react-agent', PERIODIC_CURSOR));
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, /Since the last periodic analysis/);
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date(PERIODIC_CURSOR).toISOString());
  });

  it('blocks the run when a refetch loses the picked cursor', async () => {
    server.use(analysisRunStatusHandler('react-agent', PERIODIC_CURSOR));
    const { user, dialog } = await openModal();
    await pickPreset(user, dialog, /Since the last periodic analysis/);

    server.use(
      http.get(
        mockApiUrl(getInsightsGetStatusesAnalysisRunStatusQueryKey, ':workspace', ':agent'),
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
    expect(dialog.getByRole('button', { name: 'Run insight analysis' })).toBeDisabled();
  });

  it('hides a cached cursor until the reopened modal refetches it', async () => {
    const newerCursor = '2026-08-15T09:00:00Z';
    server.use(analysisRunStatusHandler('react-agent', PERIODIC_CURSOR));
    const { user, dialog } = await openModal();
    await user.click(dialog.getByRole('combobox', { name: 'Traces to analyze' }));
    await screen.findByRole('option', { name: /Since the last periodic analysis/ });
    await user.keyboard('{Escape}');
    await user.click(dialog.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    let releaseStatus = () => {};
    const statusGate = new Promise<void>((resolve) => {
      releaseStatus = resolve;
    });
    server.use(
      http.get(
        mockApiUrl(getInsightsGetStatusesAnalysisRunStatusQueryKey, ':workspace', ':agent'),
        async () => {
          await statusGate;
          return HttpResponse.json({
            id: 'insights-analysis-run-status-react-agent',
            name: 'react-agent',
            agent: 'react-agent',
            status: 'idle',
            last_successful_run_at: newerCursor,
          });
        }
      )
    );
    await user.click(screen.getByRole('button', { name: 'Run analysis now' }));
    const reopened = within(await screen.findByRole('dialog'));
    await user.click(reopened.getByRole('combobox', { name: 'Traces to analyze' }));
    await screen.findByRole('option', { name: 'Last 24 hours' });
    expect(
      screen.queryByRole('option', { name: /Since the last periodic analysis/ })
    ).not.toBeInTheDocument();
    await user.keyboard('{Escape}');

    releaseStatus();
    await pickPreset(user, reopened, /Since the last periodic analysis/);
    const body = await submit(user, reopened);

    expect(body.since).toBe(new Date(newerCursor).toISOString());
  });

  it('does not offer the cursor before the scheduler has run', async () => {
    const { user, dialog } = await openModal();

    await user.click(dialog.getByRole('combobox', { name: 'Traces to analyze' }));
    await screen.findByRole('option', { name: 'Last 24 hours' });
    expect(
      screen.queryByRole('option', { name: /Since the last periodic analysis/ })
    ).not.toBeInTheDocument();
  });

  it('sends a day back for Last 24 hours', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Last 24 hours');
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - DAY_MS);
  });

  it('sends a week back for Last 7 days', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Last 7 days');
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - 7 * DAY_MS);
  });

  it('sends a custom lower bound read in local time', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Custom');
    const input = dialog.getByLabelText('Analyze traces since', { selector: 'input' });
    await user.clear(input);
    await user.type(input, '2026-10-01T08:30');
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date('2026-10-01T08:30').toISOString());
  });

  it('refuses a custom lower bound in the future', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Custom');
    const input = dialog.getByLabelText('Analyze traces since', { selector: 'input' });
    await user.clear(input);
    await user.type(input, `${new Date().getFullYear() + 1}-01-01T00:00`);

    expect(dialog.getByText('Enter a date and time in the past.')).toBeInTheDocument();
    expect(dialog.getByRole('button', { name: 'Run insight analysis' })).toBeDisabled();
  });

  it('explains why it cannot run when the stored config has no fast model', async () => {
    server.use(
      http.get(mockApiUrl(getInsightsGetAnalysisConfigQueryKey, ':workspace', ':agent'), () =>
        HttpResponse.json({ ...mockAnalysisConfig, fast_model: undefined })
      )
    );
    const { user, dialog } = await openModal();

    expect(
      dialog.getByText('The analysis config has no stored fast model. Pick one for this run.')
    ).toBeInTheDocument();
    expect(dialog.getByRole('button', { name: 'Run insight analysis' })).toBeDisabled();

    await user.type(dialog.getByLabelText('Fast model'), 'default/override-model');
    const body = await submit(user, dialog);

    expect(body.fast_model).toBe('default/override-model');
  });

  it('sends overridden models for this run without saving them to the config', async () => {
    const { user, dialog } = await openModal();

    await user.type(dialog.getByLabelText('Default model'), 'default/override-model');
    const body = await submit(user, dialog);

    expect(body.default_model).toBe('default/override-model');
    expect(body.fast_model).toBe(mockAnalysisConfig.fast_model);
    expect(configWrites).toEqual([]);
    expect(screen.getAllByText(mockAnalysisConfig.default_model).length).toBeGreaterThan(0);
  });

  it('returns an overridden model to the stored default with Use default', async () => {
    const { user, dialog } = await openModal();

    await user.type(dialog.getByLabelText('Fast model'), 'default/override-model');
    await user.click(dialog.getByRole('button', { name: 'Use default' }));
    expect(dialog.getByLabelText('Fast model')).toHaveValue('');
    const body = await submit(user, dialog);

    expect(body.fast_model).toBe(mockAnalysisConfig.fast_model);
  });

  it('omits the Ethos when the agent has no ETHOS.md', async () => {
    server.use(...agentEthosHandlers('react-agent', null));
    const { user, dialog } = await openModal();

    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('ethos');
  });

  it('rereads the evaluations each time it opens, so a just-indexed one appears', async () => {
    server.use(agentEvaluationsHandler('react-agent', []));
    const { user, dialog } = await openModal({ productionCache: true });

    expect(
      await dialog.findByText('No evaluations have recorded traces for this agent.')
    ).toBeInTheDocument();
    await user.click(dialog.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

    server.use(agentEvaluationsHandler('react-agent', ['baseline-3']));
    await user.click(screen.getByRole('button', { name: 'Run analysis now' }));
    const reopened = within(await screen.findByRole('dialog'));

    await waitFor(() =>
      expect(reopened.getByRole('combobox', { name: 'Scope to an evaluation' })).toBeEnabled()
    );
    await user.click(reopened.getByRole('combobox', { name: 'Scope to an evaluation' }));
    expect(await screen.findByRole('option', { name: 'baseline-3' })).toBeInTheDocument();
  });

  it("scopes the run to one of the agent's evaluations by name", async () => {
    server.use(agentEvaluationsHandler('react-agent', ['baseline-1', 'baseline-2']));
    const { user, dialog } = await openModal();

    await user.click(dialog.getByRole('combobox', { name: 'Scope to an evaluation' }));
    await user.click(await screen.findByRole('option', { name: 'baseline-2' }));
    const body = await submit(user, dialog);

    expect(body.evaluation_id).toBe('baseline-2');
  });

  it('closes after the run is queued', async () => {
    const { user, dialog } = await openModal();

    await submit(user, dialog);

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(await screen.findByText('Queued analysis run "analysis-run-1".')).toBeInTheDocument();
  });
});
