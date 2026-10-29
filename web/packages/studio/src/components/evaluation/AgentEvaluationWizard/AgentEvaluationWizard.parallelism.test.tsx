// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  createExperiment,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse, HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { fetchEvaluatorJobs } from '@studio/api/evaluation/evaluator-jobs';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import {
  createRunEvaluation,
  EVAL_CONFIG_FILESET_KEY,
  evaluationConfigError,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import { ROUTES } from '@studio/constants/routes';
import { agentDeployment, agentDeploymentsHandler } from '@studio/mocks/handlers/agentDeployments';
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
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesCreateFileset: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDownloadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/api/evaluation/evaluator-jobs', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/api/evaluation/evaluator-jobs')>()),
  fetchEvaluatorJobs: vi.fn(),
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
    agentDeploymentsHandler([agentDeployment(AGENT, `${AGENT}-subprocess`)]),
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

const setParallelism = async (user: ReturnType<typeof userEvent.setup>, value: string) => {
  const field = await screen.findByRole('spinbutton', { name: 'Parallel requests' });
  await user.clear(field);
  await user.type(field, value);
};

const submittedParallelism = () =>
  vi.mocked(evalsCreateEvaluateJob).mock.calls[0]?.[1]?.spec?.params?.parallelism;

beforeEach(() => {
  mockLists();
  vi.mocked(evalsCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(createExperiment).mockResolvedValue({
    id: 'grp_new',
    name: 'model-update-tests',
  } as never);
  vi.mocked(createRunEvaluation).mockResolvedValue('eval_new' as never);
  vi.mocked(evaluationConfigError).mockResolvedValue(null);
  vi.mocked(findEvalConfigFile).mockResolvedValue('eval-config.yaml' as never);
  vi.mocked(filesDownloadFile).mockResolvedValue(new Blob([STORED_EVAL_CONFIG]) as never);
  vi.mocked(fetchEvaluatorJobs).mockResolvedValue([]);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('AgentEvaluationWizard parallel requests', () => {
  it('submits the chosen parallelism for a new experiment', async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.upload(
      screen.getByLabelText('Dataset file'),
      new File(['{"prompt": "hi", "expected": "hi"}\n'], 'dataset.jsonl', {
        type: 'application/json',
      })
    );
    await user.upload(
      screen.getByLabelText('Eval configuration file'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );
    await setParallelism(user, '2');

    await waitFor(async () => {
      await user.click(screen.getByRole('button', { name: 'Next' }));
      expect(screen.getByLabelText('Evaluation Name')).toBeInTheDocument();
    });
    await user.type(screen.getByLabelText('Evaluation Name'), 'run-1');
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(submittedParallelism()).toBe(2);
  });

  it('submits the chosen parallelism when re-running an evaluation', async () => {
    const user = userEvent.setup();
    renderWizard();

    await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    await user.click(await screen.findByRole('option', { name: 'baseline' }));
    const name = await screen.findByLabelText('New Evaluation Name');
    await user.clear(name);
    await user.type(name, 'baseline-throttled');
    await setParallelism(user, '2');

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled()
    );
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(submittedParallelism()).toBe(2);
  });
});

describe('AgentEvaluationWizard deployment', () => {
  const submittedUrl = () => {
    const target = vi.mocked(evalsCreateEvaluateJob).mock.calls[0]?.[1]?.spec?.target;
    return typeof target === 'object' && 'url' in target ? target.url : undefined;
  };

  const rerunBaseline = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    await user.click(await screen.findByRole('option', { name: 'baseline' }));
    const name = await screen.findByLabelText('New Evaluation Name');
    await user.clear(name);
    await user.type(name, 'baseline-on-k8s');
  };

  const submit = async (user: ReturnType<typeof userEvent.setup>) => {
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled()
    );
    await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
    await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
  };

  beforeEach(() => {
    server.use(
      agentDeploymentsHandler([
        agentDeployment(AGENT, `${AGENT}-subprocess`),
        agentDeployment(AGENT, `${AGENT}-k8s`, { deployment_mode: 'k8s' }),
        agentDeployment('other-agent', 'other-agent-subprocess'),
      ])
    );
  });

  it('calls the first running deployment unless another is picked', async () => {
    const user = userEvent.setup();
    renderWizard();

    await rerunBaseline(user);
    expect(screen.getByRole('combobox', { name: 'Deployment' })).toHaveTextContent(
      `${AGENT}-subprocess · Subprocess`
    );
    await submit(user);

    expect(submittedUrl()).toContain(
      `/workspaces/${DEFAULT_WORKSPACE}/deployments/${AGENT}-subprocess/-/v1/chat/completions`
    );
  });

  it('re-runs against the deployment the source run called', async () => {
    const sourceJob = {
      name: 'baseline-job',
      workspace: DEFAULT_WORKSPACE,
      spec: {
        publication: { intake: { evaluation_id: 'baseline' } },
        target: {
          url: `https://helix.example.com/apis/agents/v2/workspaces/${DEFAULT_WORKSPACE}/deployments/${AGENT}-k8s/-/v1/chat/completions`,
        },
      },
    } as unknown as HelixJobResponse;
    vi.mocked(fetchEvaluatorJobs).mockResolvedValue([sourceJob]);
    const user = userEvent.setup();
    renderWizard();

    await rerunBaseline(user);
    await waitFor(() =>
      expect(screen.getByRole('combobox', { name: 'Deployment' })).toHaveTextContent(
        `${AGENT}-k8s · Kubernetes`
      )
    );
    await submit(user);

    expect(submittedUrl()).toContain(`/deployments/${AGENT}-k8s/-/`);
  });

  it('calls the deployment the user picks', async () => {
    const user = userEvent.setup();
    renderWizard();

    await rerunBaseline(user);
    await user.click(screen.getByRole('combobox', { name: 'Deployment' }));
    expect(screen.queryByRole('option', { name: /other-agent/ })).not.toBeInTheDocument();
    await user.click(await screen.findByRole('option', { name: `${AGENT}-k8s · Kubernetes` }));
    await submit(user);

    expect(submittedUrl()).toContain(`/deployments/${AGENT}-k8s/-/`);
  });

  it('holds the first step when the agent has no running deployment', async () => {
    server.use(
      agentDeploymentsHandler([agentDeployment(AGENT, `${AGENT}-down`, { status: 'failed' })])
    );
    renderWizard();

    expect(
      await screen.findByText(`${AGENT} has no running deployment. Deploy it to run an evaluation.`)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });
});
