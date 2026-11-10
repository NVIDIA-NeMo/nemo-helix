// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  createExperiment,
  getExperiment,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import { filesCreateFileset } from '@nemo/sdk/generated/platform/files';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { DEFAULT_EXPERIMENT_AGENT_KEY } from '@studio/components/evaluation/defaultExperiment';
import {
  createRunEvaluation,
  EVAL_CONFIG_FILESET_KEY,
  recordRunJob,
} from '@studio/components/evaluation/experimentEvalConfig';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@nemo/sdk/generated/evals/evals-plugin-jobs-routes', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/evals/evals-plugin-jobs-routes')>()),
  evalsCreateEvaluateJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/experiments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/experiments')>()),
  createExperiment: vi.fn(),
  getExperiment: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesCreateFileset: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
  recordRunJob: vi.fn(),
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

const mockLists = () => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [
          {
            id: 'grp_primary',
            name: 'primary-use-cases-benchmark',
            workspace: DEFAULT_WORKSPACE,
            default_sort: '-created_at',
            evaluation_count: 1,
          },
        ],
      })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
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
      { path: ROUTES.workspace.agentDetail, element: <div>Agent page</div> },
    ],
  });

const pickFiles = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.upload(
    await screen.findByLabelText('Dataset file'),
    new File(['{"prompt": "hi", "expected": "hi"}\n'], 'dataset.jsonl', {
      type: 'application/json',
    })
  );
  await user.upload(
    screen.getByLabelText('Eval configuration file'),
    new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
  );
};

/** Run Evaluation is retried because the picked files are read asynchronously. */
const run = async (user: ReturnType<typeof userEvent.setup>) =>
  waitFor(async () => {
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
    expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1);
  });

const notFound = Object.assign(new Error('Not Found'), { response: { status: 404 } });

beforeEach(() => {
  mockLists();
  vi.mocked(evalsCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(getExperiment).mockRejectedValue(notFound);
  vi.mocked(createExperiment).mockResolvedValue({ id: 'grp_default' } as never);
  vi.mocked(createRunEvaluation).mockImplementation(async (_ws, { name }) => name);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('AgentEvaluationWizard submit', () => {
  it("runs with only a dataset and config, landing in the agent's default experiment", async () => {
    const user = userEvent.setup();
    renderWizard();

    await pickFiles(user);
    await run(user);

    expect(createExperiment).toHaveBeenCalledWith(
      DEFAULT_WORKSPACE,
      expect.objectContaining({
        name: 'my-agent-runs',
        metadata: { [DEFAULT_EXPERIMENT_AGENT_KEY]: AGENT },
      }),
      undefined
    );
    const [, { name, experimentIds, filesetName }] = vi.mocked(createRunEvaluation).mock.calls[0];
    expect(experimentIds).toEqual(['grp_default']);
    expect(name).toMatch(/^my-agent-[a-z0-9]{8}$/);
    expect(filesetName).toBe(`${name}-data`);
    expect(filesCreateFileset).toHaveBeenCalledWith(
      DEFAULT_WORKSPACE,
      expect.objectContaining({ name: `${name}-data` })
    );
    const [, request] = vi.mocked(evalsCreateEvaluateJob).mock.calls[0];
    expect(request.spec).toMatchObject({ publication: { intake: { evaluation_id: name } } });
    expect(recordRunJob).toHaveBeenCalledWith(DEFAULT_WORKSPACE, name, 'job-1', {
      [EVAL_CONFIG_FILESET_KEY]: `${name}-data`,
    });
  });

  it('reuses an existing default experiment instead of creating another', async () => {
    vi.mocked(getExperiment).mockResolvedValue({ id: 'grp_existing_default' } as never);
    const user = userEvent.setup();
    renderWizard();

    await pickFiles(user);
    await run(user);

    expect(createExperiment).not.toHaveBeenCalled();
    expect(vi.mocked(createRunEvaluation).mock.calls[0][1].experimentIds).toEqual([
      'grp_existing_default',
    ]);
  });

  it('links the run to a picked experiment and names it as typed', async () => {
    const user = userEvent.setup();
    renderWizard();

    await pickFiles(user);
    await user.click(screen.getByRole('combobox', { name: /^Experiment/ }));
    await user.click(await screen.findByRole('option', { name: 'primary-use-cases-benchmark' }));
    await user.type(screen.getByLabelText('Evaluation Name (Optional)'), 'run-1');
    await run(user);

    expect(getExperiment).not.toHaveBeenCalled();
    expect(createRunEvaluation).toHaveBeenCalledWith(DEFAULT_WORKSPACE, {
      experimentIds: ['grp_primary'],
      name: 'run-1',
      filesetName: 'run-1-data',
    });
  });

  it('submits the chosen parallelism', async () => {
    const user = userEvent.setup();
    renderWizard();

    await pickFiles(user);
    const field = screen.getByRole('spinbutton', { name: 'Parallel requests' });
    await user.clear(field);
    await user.type(field, '2');
    await run(user);

    const [, request] = vi.mocked(evalsCreateEvaluateJob).mock.calls[0];
    expect(request.spec?.params?.parallelism).toBe(2);
  });
});
