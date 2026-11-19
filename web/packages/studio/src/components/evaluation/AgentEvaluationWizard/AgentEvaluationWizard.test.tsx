// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { getFilesListFilesetFilesQueryKey } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { DEFAULT_EXPERIMENT_AGENT_KEY } from '@studio/components/evaluation/defaultExperiment';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer: () => async () =>
    new TextEncoder().encode('{"prompt": "hi", "expected": "hi"}\n').buffer,
}));

vi.mock('@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect', async () =>
  (await import('@studio/tests/util/mockDatasetPick')).datasetFileSelectMock()
);

vi.mock('@studio/components/evaluation/EvalWizard/useDatasetPreview', async (importOriginal) =>
  (await import('@studio/tests/util/mockDatasetPick')).datasetPreviewMock(await importOriginal(), {
    prompt: 'hi',
    expected: 'hi',
  })
);

const AGENT = 'my-agent';

/** A minimal eval config: parsed locally, never submitted. */
const EVAL_CONFIG = `prompt_template: "{{ item.prompt }}"
metrics:
  - bundle_kind: metric-bundle
    bundle_format_version: v1
    metric_type: string-check
    metadata: {}
    outputs:
      - name: string-check
        value_json_schema:
          type: number
    secrets: {}
    payload:
      kind: inline
      metric:
        type: string-check
        operation: equals
        left_template: "{{ sample.output_text }}"
        right_template: "{{ item.expected }}"
`;

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
    }),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ data: [] })
    )
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

type User = ReturnType<typeof userEvent.setup>;

/** Configure a dataset and an uploaded eval config, then move on to the Run step. */
const openRunStep = async (user: User) => {
  await user.click(await screen.findByLabelText('Dataset file ref'));
  await user.paste(`${DEFAULT_WORKSPACE}/data#rows.jsonl`);
  await user.click(
    within(await screen.findByRole('radiogroup', { name: 'Eval configuration source' })).getByRole(
      'radio',
      { name: 'Upload' }
    )
  );
  await user.upload(
    screen.getByLabelText('Eval configuration file'),
    new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
  );
  await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
  await user.click(screen.getByRole('button', { name: 'Next' }));
  await screen.findByRole('button', { name: 'Run Evaluation' });
};

describe('AgentEvaluationWizard', () => {
  beforeEach(mockLists);

  it('opens on Configuration, with only Run after it', async () => {
    renderWizard();

    expect(await screen.findByLabelText('Dataset file ref')).toBeInTheDocument();
    expect(screen.getByText('Configuration')).toBeInTheDocument();
    expect(screen.getByText('Run')).toBeInTheDocument();
    expect(screen.queryByText('How do you want to start?')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next' })).toBeInTheDocument();
  });

  it('asks nothing on the Run step that has to be filled in', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openRunStep(user);

    expect(screen.getByLabelText('Evaluation Name (Optional)')).toHaveValue('');
    expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled();
  });

  it("offers real experiments to link, but not the agent's default one", async () => {
    const user = userEvent.setup();
    renderWizard();

    await openRunStep(user);
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
    const user = userEvent.setup();
    renderWizard();

    await openRunStep(user);

    expect(screen.getByRole('spinbutton', { name: 'Parallel Requests' })).toHaveValue(1);
  });

  it('holds Run Evaluation on a taken name, saying so only under the name field', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openRunStep(user);
    await user.type(screen.getByLabelText('Evaluation Name (Optional)'), 'baseline');

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
