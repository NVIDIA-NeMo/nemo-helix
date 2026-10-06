// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evaluatorCreateEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  createExperiment,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import {
  filesUploadFile,
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { createRunEvaluation } from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { act, renderRoute, screen, waitFor } from '@studio/tests/util/render';
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
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/api/datasets/useDatasetFileContent', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/api/datasets/useDatasetFileContent')>()),
  datasetFileContentQueryOptions: vi.fn(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
}));

const AGENT = 'my-agent';
const ROWS = '{"prompt": "hi", "expected": "hi"}\n{"prompt": "yo", "expected": "yo"}\n';

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

const mockListApis = () => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: [] })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: [] })
    ),
    http.get(mockApiUrl(getFilesListFilesetsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: 'generated', workspace: DEFAULT_WORKSPACE }],
        pagination: { total: 1, page: 1, page_size: 20 },
      })
    ),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({
        data: [
          {
            file_ref: `${DEFAULT_WORKSPACE}/generated#output/part-0.parquet`,
            file_url: '/output/part-0.parquet',
            path: 'output/part-0.parquet',
            size: 10,
          },
        ],
      })
    )
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

beforeEach(() => {
  mockListApis();
  vi.mocked(datasetFileContentQueryOptions).mockImplementation(
    ({ path }) => ({ queryKey: ['test-file-content', path], queryFn: async () => ROWS }) as never
  );
  vi.mocked(evaluatorCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(createExperiment).mockResolvedValue({
    id: 'grp_new',
    name: 'model-update-tests',
  } as never);
  vi.mocked(createRunEvaluation).mockResolvedValue('eval_new' as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('SubmitEvaluationModal dataset from a fileset', () => {
  it('stores the picked fileset file as the run dataset and submits it', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
    await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
    await user.click(await screen.findByRole('option', { name: 'generated' }));
    await user.click(await screen.findByRole('combobox', { name: 'File' }));
    await user.click(await screen.findByRole('option', { name: 'output/part-0.parquet' }));
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    const [, spec] = vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0];
    expect(spec.spec).toMatchObject({ dataset: expect.stringMatching(/#dataset\.jsonl$/) });
    const datasetUpload = vi
      .mocked(filesUploadFile)
      .mock.calls.find(([, , name]) => name === 'dataset.jsonl');
    expect(await (datasetUpload?.[3] as File).text()).toBe(ROWS);
  });

  it('drops a fileset read that finishes after switching back to upload', async () => {
    let finishRead: (text: string) => void = () => {};
    vi.mocked(datasetFileContentQueryOptions).mockImplementation(
      ({ path }) =>
        ({
          queryKey: ['test-file-content', path],
          queryFn: () => new Promise<string>((resolve) => (finishRead = resolve)),
        }) as never
    );
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
    await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
    await user.click(await screen.findByRole('option', { name: 'generated' }));
    await user.click(await screen.findByRole('combobox', { name: 'File' }));
    await user.click(await screen.findByRole('option', { name: 'output/part-0.parquet' }));
    await user.click(screen.getByRole('radio', { name: 'Upload a file' }));
    await act(async () => {
      finishRead(ROWS);
      await new Promise((resolve) => setTimeout(resolve, 50));
    });

    expect(screen.getByLabelText('Add Dataset')).toBeInTheDocument();
  });

  it('keeps Submit from sending a fileset dataset that was never picked', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    expect(await screen.findByText('Add a dataset')).toBeVisible();
    expect(evaluatorCreateEvaluateJob).not.toHaveBeenCalled();
  });
});
