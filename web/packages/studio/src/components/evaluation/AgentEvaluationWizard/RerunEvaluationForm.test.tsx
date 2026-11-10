// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import {
  deleteEvaluation,
  getEvaluation,
  getListEvaluationsQueryKey,
} from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import { submitAgentEvalJob } from '@studio/api/evaluation/agent-evaluations';
import { RerunEvaluationForm } from '@studio/components/evaluation/AgentEvaluationWizard/RerunEvaluationForm';
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

vi.mock('@studio/api/evaluation/agent-evaluations', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/api/evaluation/agent-evaluations')>()),
  submitAgentEvalJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/jobs', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/jobs')>()),
  jobsGetJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/evaluations', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/evaluations')>()),
  getEvaluation: vi.fn(),
  deleteEvaluation: vi.fn(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
  recordRunJob: vi.fn(),
}));

const AGENT = 'my-agent';
const SOURCE_JOB = 'baseline-abc12345';

const DATASET_SPEC = {
  dataset: 'default/baseline-data#dataset.jsonl',
  metrics: [{ metric_type: 'string-check' }],
  prompt_template: '{{ item.prompt }}',
  target: { name: AGENT },
  params: { parallelism: 4 },
  publication: { intake: { evaluation_id: 'baseline' } },
};

const renderForm = () =>
  renderRoute(undefined, {
    history: `/workspaces/${DEFAULT_WORKSPACE}`,
    routes: [
      {
        path: '/workspaces/:workspace',
        element: (
          <RerunEvaluationForm workspace={DEFAULT_WORKSPACE} agent={AGENT} sourceJob={SOURCE_JOB} />
        ),
      },
      { path: ROUTES.workspace.agentDetail, element: <div>Agent page</div> },
    ],
  });

beforeEach(() => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [
          { id: 'grp_primary', name: 'primary-use-cases-benchmark', workspace: DEFAULT_WORKSPACE },
        ],
      })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: [] })
    )
  );
  vi.mocked(jobsGetJob).mockResolvedValue({
    name: SOURCE_JOB,
    workspace: DEFAULT_WORKSPACE,
    spec: DATASET_SPEC,
  } as never);
  vi.mocked(getEvaluation).mockResolvedValue({
    id: 'eval_baseline',
    name: 'baseline',
    experiment_ids: ['grp_primary'],
    metadata: { [EVAL_CONFIG_FILESET_KEY]: 'baseline-data' },
  } as never);
  vi.mocked(createRunEvaluation).mockImplementation(async (_ws, { name }) => name);
  vi.mocked(evalsCreateEvaluateJob).mockResolvedValue({ name: 'job-2' } as never);
  vi.mocked(submitAgentEvalJob).mockResolvedValue({ name: 'job-2' } as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('RerunEvaluationForm', () => {
  it("defaults to the source run's experiment and submits a copy of its spec", async () => {
    const user = userEvent.setup();
    renderForm();

    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: /^Experiment/ })).toHaveTextContent(
        'primary-use-cases-benchmark'
      )
    );
    await user.type(screen.getByLabelText('Evaluation Name (Optional)'), 'baseline-again');
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled()
    );
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(createRunEvaluation).toHaveBeenCalledWith(DEFAULT_WORKSPACE, {
      experimentIds: ['grp_primary'],
      name: 'baseline-again',
      filesetName: 'baseline-data',
      parentEvaluationId: 'eval_baseline',
    });
    const [, request] = vi.mocked(evalsCreateEvaluateJob).mock.calls[0];
    expect(request.spec).toEqual({
      ...DATASET_SPEC,
      publication: { intake: { evaluation_id: 'baseline-again' } },
    });
    expect(recordRunJob).toHaveBeenCalledWith(DEFAULT_WORKSPACE, 'baseline-again', 'job-2', {
      [EVAL_CONFIG_FILESET_KEY]: 'baseline-data',
    });
  });

  it('re-submits a task-based job to the agent-evaluate endpoint', async () => {
    vi.mocked(jobsGetJob).mockResolvedValue({
      name: SOURCE_JOB,
      workspace: DEFAULT_WORKSPACE,
      spec: { tasks: [{ id: 't1' }], target: { kind: 'agent' } },
    } as never);
    vi.mocked(getEvaluation).mockRejectedValue(new Error('gone'));
    const user = userEvent.setup();
    renderForm();

    await user.type(await screen.findByLabelText('Evaluation Name (Optional)'), 'task-rerun');
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled()
    );
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    await waitFor(() => expect(submitAgentEvalJob).toHaveBeenCalledTimes(1));
    expect(evalsCreateEvaluateJob).not.toHaveBeenCalled();
  });

  it('removes the new evaluation when the job cannot be submitted', async () => {
    vi.mocked(evalsCreateEvaluateJob).mockRejectedValue(new Error('spec rejected'));
    vi.mocked(deleteEvaluation).mockResolvedValue(undefined as never);
    const user = userEvent.setup();
    renderForm();

    await user.type(await screen.findByLabelText('Evaluation Name (Optional)'), 'doomed');
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled()
    );
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    expect(await screen.findByText('spec rejected')).toBeVisible();
    expect(deleteEvaluation).toHaveBeenCalledWith(DEFAULT_WORKSPACE, 'doomed');
  });
});
