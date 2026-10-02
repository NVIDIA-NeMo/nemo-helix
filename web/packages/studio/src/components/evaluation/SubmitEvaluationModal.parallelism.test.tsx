// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evaluatorCreateEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  createExperiment,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  createRunEvaluation,
  EVAL_CONFIG_FILESET_KEY,
  evaluationConfigError,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes', async (importOriginal) => ({
  ...(await importOriginal<
    typeof import('@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes')
  >()),
  evaluatorCreateEvaluateJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/experiments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/experiments')>()),
  createExperiment: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesCreateFileset: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDownloadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
  evaluationConfigError: vi.fn(),
  findEvalConfigFile: vi.fn(),
}));

const AGENT = 'my-agent';

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

// A stored config carries the dataset reference that upload bakes in.
const STORED_EVAL_CONFIG = `dataset: default/baseline-data#dataset.jsonl
${EVAL_CONFIG}`;

const SOURCE: EvaluationResponse = {
  id: 'eval_baseline',
  name: 'baseline',
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: ['grp_primary'],
  experiment_group_id: 'grp_primary',
  dataset_name: 'ds',
  agent_names: [AGENT],
  metadata: { [EVAL_CONFIG_FILESET_KEY]: 'baseline-data' },
};

const mockLists = () => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), ({ request }) => {
      const name = new URL(request.url).searchParams.get('filter[name]');
      const experiments = [
        {
          id: 'grp_primary',
          name: 'primary-use-cases-benchmark',
          workspace: DEFAULT_WORKSPACE,
          default_sort: '-created_at',
          evaluation_count: 1,
        },
      ];
      return HttpResponse.json({ data: name ? [] : experiments });
    }),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), ({ request }) => {
      const name = new URL(request.url).searchParams.get('filter[name]');
      return HttpResponse.json({ data: name ? [] : [SOURCE] });
    })
  );
};

const renderModal = () =>
  renderRoute(undefined, {
    history: `/workspaces/${DEFAULT_WORKSPACE}`,
    routes: [
      {
        path: '/workspaces/:workspace',
        element: (
          <SubmitEvaluationModal
            open
            onClose={() => {}}
            workspace={DEFAULT_WORKSPACE}
            agent={AGENT}
          />
        ),
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent page</div> },
    ],
  });

const setParallelism = async (user: ReturnType<typeof userEvent.setup>, value: string) => {
  const field = await screen.findByRole('spinbutton', { name: 'Parallel requests' });
  await user.clear(field);
  await user.type(field, value);
};

const submittedParallelism = () =>
  vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0]?.[1]?.spec?.params?.parallelism;

beforeEach(() => {
  mockLists();
  vi.mocked(evaluatorCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(createExperiment).mockResolvedValue({
    id: 'grp_new',
    name: 'model-update-tests',
  } as never);
  vi.mocked(createRunEvaluation).mockResolvedValue('eval_new' as never);
  vi.mocked(evaluationConfigError).mockResolvedValue(null);
  vi.mocked(findEvalConfigFile).mockResolvedValue('eval-config.yaml' as never);
  vi.mocked(filesDownloadFile).mockResolvedValue(new Blob([STORED_EVAL_CONFIG]) as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('SubmitEvaluationModal parallel requests', () => {
  it('submits the chosen parallelism for a new experiment', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.upload(
      screen.getByLabelText('Add Dataset'),
      new File(['{"prompt": "hi", "expected": "hi"}\n'], 'dataset.jsonl', {
        type: 'application/json',
      })
    );
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );
    await setParallelism(user, '2');

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(submittedParallelism()).toBe(2);
  });

  it('submits the chosen parallelism when re-running an evaluation', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    await user.click(await screen.findByRole('option', { name: 'baseline' }));
    const name = await screen.findByLabelText('New Evaluation Name');
    await user.clear(name);
    await user.type(name, 'baseline-throttled');
    await setParallelism(user, '2');

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(submittedParallelism()).toBe(2);
  });
});
