// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { SwitchyardOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm';
import { useSubmitSwitchyardOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/useSubmitSwitchyardOptimization';
import { getAgentOptimizeRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

// The real picker pages through the model catalogue; a button per model keeps these tests on the form.
vi.mock('@nemo/common/src/components/ModelSelectV2', () => ({
  WorkspaceModelSelect: ({
    onValueChange,
    'aria-label': label,
  }: {
    onValueChange: (selection: { model: string }) => void;
    'aria-label'?: string;
  }) => (
    <div aria-label={label} role="group">
      {['default/big', 'default/medium', 'default/small'].map((model) => (
        <button key={model} type="button" onClick={() => onValueChange({ model })}>
          {model}
        </button>
      ))}
    </div>
  ),
}));

const AGENT = 'react-agent';
const JOBS_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy`;

const SubmitForm = () => {
  const onSubmit = useSubmitSwitchyardOptimization({
    workspace: DEFAULT_WORKSPACE,
    agentName: AGENT,
  });
  return <SwitchyardOptimizationForm agentName={AGENT} onBack={() => {}} onSubmit={onSubmit} />;
};

const renderForm = () =>
  renderRoute(undefined, {
    history: getAgentOptimizeRoute(DEFAULT_WORKSPACE, AGENT),
    routes: [{ path: ROUTES.workspace.agentDetail, element: <SubmitForm /> }],
  });

const addModel = async (user: ReturnType<typeof userEvent.setup>, model: string) =>
  user.click(within(screen.getByRole('group', { name: 'Add a model' })).getByText(model));

describe('SwitchyardOptimizationForm', () => {
  it('holds the run closed until there are two models to route between', async () => {
    const user = userEvent.setup();
    renderForm();

    const run = await screen.findByRole('button', { name: 'Run optimization' });
    expect(run).toBeDisabled();
    expect(screen.getByText('Add at least two models to route between.')).toBeInTheDocument();

    await addModel(user, 'default/big');
    expect(run).toBeDisabled();

    await addModel(user, 'default/small');
    await waitFor(() => expect(run).toBeEnabled());
  });

  it('reorders and removes models', async () => {
    const user = userEvent.setup();
    renderForm();

    await addModel(user, 'default/big');
    await addModel(user, 'default/small');
    await user.click(screen.getByRole('button', { name: 'Move default/small up' }));

    const list = screen.getByRole('list', { name: 'Models, most capable first' });
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent)
    ).toEqual([expect.stringContaining('default/small'), expect.stringContaining('default/big')]);

    await user.click(screen.getByRole('button', { name: 'Remove default/big' }));
    expect(within(list).getAllByRole('listitem')).toHaveLength(1);
  });

  it('opens a strategy’s settings inside its card only once it is picked', async () => {
    const user = userEvent.setup();
    renderForm();

    expect(await screen.findByText('Capable model share')).toBeInTheDocument();
    expect(screen.queryByText('Confidence threshold')).not.toBeInTheDocument();
    expect(screen.queryByText('Judge model (optional)')).not.toBeInTheDocument();

    await user.click(screen.getByRole('checkbox', { name: /LLM classifier/ }));

    expect(await screen.findByText('Capability threshold')).toBeInTheDocument();
    expect(screen.getByText('Judge model (optional)')).toBeInTheDocument();
  });

  it('lists every pair the run will route, capable model first', async () => {
    const user = userEvent.setup();
    renderForm();

    await addModel(user, 'default/big');
    await addModel(user, 'default/medium');
    await addModel(user, 'default/small');

    const pairs = screen.getByRole('list', { name: 'Model pairs' });
    expect(
      within(pairs)
        .getAllByRole('listitem')
        .map((item) => item.textContent)
    ).toEqual([
      'bigcapablemediumefficient',
      'bigcapablesmallefficient',
      'mediumcapablesmallefficient',
    ]);
    // The workspace prefix is dropped from the label but kept as the tooltip.
    expect(within(pairs).getAllByTitle('default/small')).toHaveLength(2);
  });

  it('blocks the run when no routing strategy is picked', async () => {
    const user = userEvent.setup();
    renderForm();

    await addModel(user, 'default/big');
    await addModel(user, 'default/small');
    await user.click(screen.getByRole('checkbox', { name: /Random routing/ }));

    expect(await screen.findByText('Pick at least one routing strategy.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run optimization' })).toBeDisabled();
  });

  it('submits a switchyard run-strategy job with the routed models and strategies', async () => {
    const submitted: unknown[] = [];
    server.use(
      http.post(JOBS_URL, async ({ request }) => {
        submitted.push(await request.json());
        return HttpResponse.json({ name: 'routing-1' });
      })
    );
    const user = userEvent.setup();
    renderForm();

    await addModel(user, 'default/big');
    await addModel(user, 'default/medium');
    await addModel(user, 'default/small');
    await user.click(screen.getByRole('checkbox', { name: /Stage router/ }));

    expect(screen.getByText('3 pairs × 2 strategies.', { exact: false })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Run optimization' }));

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]).toMatchObject({
      name: expect.stringMatching(/^react-agent-routing-/),
      spec: {
        strategy: 'switchyard',
        agent: AGENT,
        workspace: DEFAULT_WORKSPACE,
        models: ['default/big', 'default/medium', 'default/small'],
        routing_strategies: ['random_routing', 'stage_router'],
        strong_probability: 0.5,
        confidence_threshold: 0.5,
      },
    });
    const { spec } = submitted[0] as { spec: Record<string, unknown> };
    expect(spec).not.toHaveProperty('base_threshold');
    expect(spec).not.toHaveProperty('judge_model');
  });

  it('shows the server’s rejection beside the run button', async () => {
    server.use(
      http.post(JOBS_URL, () =>
        HttpResponse.json({ detail: 'Model default/big not found.' }, { status: 422 })
      )
    );
    const user = userEvent.setup();
    renderForm();

    await addModel(user, 'default/big');
    await addModel(user, 'default/small');
    await user.click(screen.getByRole('button', { name: 'Run optimization' }));

    expect(await screen.findByText(/Model default\/big not found/)).toBeInTheDocument();
  });
});
