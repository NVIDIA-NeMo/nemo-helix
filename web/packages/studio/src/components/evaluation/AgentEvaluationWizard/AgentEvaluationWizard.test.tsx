// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { DEFAULT_EXPERIMENT_AGENT_KEY } from '@studio/components/evaluation/defaultExperiment';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const AGENT = 'my-agent';

const emptyPage = { data: [], pagination: { total: 0, page: 1, page_size: 100 } };

const experiment = (id: string, name: string, metadata = {}): ExperimentResponse => ({
  id,
  name,
  workspace: DEFAULT_WORKSPACE,
  default_sort: '-created_at',
  evaluation_count: 1,
  metadata,
});

const EXPERIMENTS = [
  experiment('grp_primary', 'primary-use-cases-benchmark'),
  experiment('grp_default', 'my-agent-runs', { [DEFAULT_EXPERIMENT_AGENT_KEY]: AGENT }),
];

const TAKEN: EvaluationResponse = {
  id: 'eval_baseline',
  name: 'baseline',
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: ['grp_primary'],
  experiment_group_id: 'grp_primary',
  dataset_name: 'ds',
  agent_names: [AGENT],
};

const mockLists = () => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: EXPERIMENTS })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), ({ request }) => {
      // The only evaluation lookup left is the name-conflict probe.
      const name = new URL(request.url).searchParams.get('filter[name]');
      return HttpResponse.json({ ...emptyPage, data: name === TAKEN.name ? [TAKEN] : [] });
    })
  );
};

const renderWizard = () =>
  renderRoute(undefined, {
    history: `/workspaces/${DEFAULT_WORKSPACE}`,
    routes: [
      {
        path: '/workspaces/:workspace',
        element: <AgentEvaluationWizard workspace={DEFAULT_WORKSPACE} agent={AGENT} />,
      },
      { path: ROUTES.workspace.agentDetail, element: <LocationDisplay /> },
    ],
  });

describe('AgentEvaluationWizard', () => {
  beforeEach(mockLists);

  it('opens straight on the dataset and eval config, with no steps to walk through', async () => {
    renderWizard();

    expect(await screen.findByLabelText('Dataset file')).toBeInTheDocument();
    expect(screen.getByLabelText('Eval configuration file')).toBeInTheDocument();
    expect(screen.queryByText('How do you want to start?')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Next' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled();
  });

  it('asks only for the dataset and eval config before running', async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.click(await screen.findByRole('button', { name: 'Run Evaluation' }));

    expect(await screen.findByText('Add a dataset')).toBeVisible();
    expect(screen.getByText('Select an eval configuration')).toBeVisible();
    // Where the results land is optional, so neither field reports anything missing.
    expect(screen.getByLabelText('Evaluation Name (Optional)')).toHaveValue('');
    expect(screen.queryByText(/Evaluation name is required/)).not.toBeInTheDocument();
  });

  it("offers real experiments to link up front, but not the agent's default one", async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.click(await screen.findByRole('combobox', { name: /^Experiment/ }));

    expect(
      await screen.findByRole('option', { name: 'primary-use-cases-benchmark' })
    ).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'my-agent-runs' })).not.toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: 'None — link later (kept in my-agent-runs)' })
    ).toBeInTheDocument();
  });

  it('offers a parallel-requests setting that defaults to 1', async () => {
    renderWizard();

    expect(await screen.findByRole('spinbutton', { name: 'Parallel requests' })).toHaveValue(1);
  });

  it('holds Run Evaluation on a taken name, saying so only under the name field', async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.type(await screen.findByLabelText('Evaluation Name (Optional)'), 'baseline');

    expect(await screen.findByText(/named baseline already exists/)).toBeInTheDocument();
    expect(screen.getAllByText(/named baseline already exists/)).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeDisabled();
  });

  it("cancels to the agent's evaluations when opened by a direct link", async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.click(await screen.findByRole('button', { name: 'Cancel' }));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getAgentEvaluationsTabRoute(DEFAULT_WORKSPACE, AGENT)
    );
  });
});
