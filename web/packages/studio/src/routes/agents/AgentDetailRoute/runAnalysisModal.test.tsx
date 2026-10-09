// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_AGENT_OVERVIEW_ENABLED', 'true');
});

import type {
  AnalysisRunResponse,
  CreateAnalysisRunRequest,
} from '@nemo/sdk/generated/insights/schema';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import {
  agentEthosHandlers,
  agentEvaluationsHandler,
  analysisRunCreateHandler,
  analysisRunHandlers,
  mockAnalysisConfig,
  mockAnalysisRunWithJob,
} from '@studio/mocks/handlers/insights';
import { server } from '@studio/mocks/node';
import { AgentDetailRoute } from '@studio/routes/agents/AgentDetailRoute';
import { DAY_MS } from '@studio/routes/agents/AgentDetailRoute/analysis/analysisSince';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    value,
    onValueChange,
    ...props
  }: {
    value: { model: string } | null;
    onValueChange: (next: { model: string }) => void;
    'aria-label': string;
  }) => (
    <input
      aria-label={props['aria-label']}
      value={value?.model ?? ''}
      onChange={(event) => onValueChange({ model: event.target.value })}
    />
  ),
}));

const LAST_RUN = mockAnalysisRunWithJob('react-agent', 'completed');

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

const openModal = async () => {
  const user = userEvent.setup();
  renderRoute(undefined, {
    history: `${getAgentDetailRoute(workspace1.workspace, 'react-agent')}?tab=insights`,
    routes: [{ path: ROUTES.workspace.agentDetail, element: <AgentDetailRoute /> }],
  });
  await user.click(await screen.findByRole('button', { name: 'Run analysis now' }));
  const dialog = within(await screen.findByRole('dialog'));
  await waitFor(() =>
    expect(dialog.queryByText('Looking for the last completed analysis run...')).toBeNull()
  );
  return { user, dialog };
};

const pickPreset = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>,
  name: string
) => {
  await user.click(dialog.getByRole('combobox', { name: 'Traces to analyze' }));
  await user.click(await screen.findByRole('option', { name }));
};

const submit = async (
  user: ReturnType<typeof userEvent.setup>,
  dialog: ReturnType<typeof within>
) => {
  const button = dialog.getByRole('button', { name: 'Run analysis' });
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
  it('opens from Run analysis now with the stored model pair and the Ethos included', async () => {
    const { dialog } = await openModal();

    await waitFor(() => expect(dialog.getByRole('button', { name: 'Run analysis' })).toBeEnabled());
    expect(dialog.getByLabelText('Default model')).toHaveValue(mockAnalysisConfig.default_model);
    expect(dialog.getByLabelText('Fast model')).toHaveValue(mockAnalysisConfig.fast_model);
    expect(await dialog.findByText('From react-agent-ethos/ETHOS.md.')).toBeInTheDocument();
    expect(dialog.getByRole('checkbox', { name: "Include the agent's Ethos" })).toBeChecked();
  });

  it('defaults to the last 24 hours when the agent has no completed run', async () => {
    const { user, dialog } = await openModal();

    expect(dialog.getByRole('combobox', { name: 'Traces to analyze' })).toHaveTextContent(
      'Last 24 hours'
    );
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - DAY_MS);
    expect(body).toMatchObject({
      agent: 'react-agent',
      default_model: mockAnalysisConfig.default_model,
      fast_model: mockAnalysisConfig.fast_model,
      ethos: '# React agent ethos',
    });
    expect(body).not.toHaveProperty('evaluation_id');
  });

  it('defaults to the start of the last completed run when there is one', async () => {
    runs = [LAST_RUN];
    const { user, dialog } = await openModal();

    expect(dialog.getByRole('combobox', { name: 'Traces to analyze' })).toHaveTextContent(
      'Since the last analysis run'
    );
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date(LAST_RUN.run.created_at).toISOString());
  });

  it('looks past newer runs that did not complete for the last completed one', async () => {
    runs = [
      mockAnalysisRunWithJob('react-agent', 'cancelled', {
        name: 'analysis-run-3',
        created_at: '2026-08-16T09:00:00Z',
      }),
      mockAnalysisRunWithJob('react-agent', 'error', {
        name: 'analysis-run-2',
        created_at: '2026-08-15T09:00:00Z',
      }),
      LAST_RUN,
    ];
    const { user, dialog } = await openModal();

    expect(dialog.getByRole('combobox', { name: 'Traces to analyze' })).toHaveTextContent(
      'Since the last analysis run'
    );
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date(LAST_RUN.run.created_at).toISOString());
  });

  it('sends a week back for Last 7 days', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Last 7 days');
    const body = await submit(user, dialog);

    expectAbout(body.since, Date.now() - 7 * DAY_MS);
  });

  it('omits since for All history', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'All history');
    expect(dialog.getByText("Analyzes the agent's full trace history.")).toBeInTheDocument();
    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('since');
  });

  it('sends a custom lower bound read in local time', async () => {
    const { user, dialog } = await openModal();

    await pickPreset(user, dialog, 'Custom');
    const input = dialog.getByLabelText('Analyze traces since');
    await user.clear(input);
    await user.type(input, '2026-10-01T08:30');
    const body = await submit(user, dialog);

    expect(body.since).toBe(new Date('2026-10-01T08:30').toISOString());
  });

  it('sends overridden models for this run without saving them to the config', async () => {
    const { user, dialog } = await openModal();

    await user.clear(dialog.getByLabelText('Default model'));
    await user.type(dialog.getByLabelText('Default model'), 'default/override-model');
    const body = await submit(user, dialog);

    expect(body.default_model).toBe('default/override-model');
    expect(body.fast_model).toBe(mockAnalysisConfig.fast_model);
    expect(configWrites).toEqual([]);
    expect(screen.getAllByText(mockAnalysisConfig.default_model).length).toBeGreaterThan(0);
  });

  it('leaves the Ethos out when unchecked', async () => {
    const { user, dialog } = await openModal();

    await user.click(
      await dialog.findByRole('checkbox', { name: "Include the agent's Ethos", checked: true })
    );
    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('ethos');
  });

  it('disables the Ethos checkbox when the agent has no ETHOS.md', async () => {
    server.use(...agentEthosHandlers('react-agent', null));
    const { user, dialog } = await openModal();

    expect(await dialog.findByText('No ETHOS.md found in react-agent-ethos.')).toBeInTheDocument();
    expect(dialog.getByRole('checkbox', { name: "Include the agent's Ethos" })).toBeDisabled();
    const body = await submit(user, dialog);

    expect(body).not.toHaveProperty('ethos');
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
