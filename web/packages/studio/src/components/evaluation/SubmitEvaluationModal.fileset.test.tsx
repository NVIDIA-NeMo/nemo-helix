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
  filesDownloadFile,
  filesUploadFile,
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { createRunEvaluation } from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import type { DownloadFileAsArrayBufferArgs } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { PARQUET, parquetFile } from '@studio/tests/util/parquetFixtures';
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
  filesDownloadFile: vi.fn(),
  filesUploadFile: vi.fn(),
  filesDeleteFileset: vi.fn(),
}));

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer:
    () =>
    async ({ workspace, datasetName, path }: DownloadFileAsArrayBufferArgs) =>
      (await filesDownloadFile(workspace, datasetName, path)).arrayBuffer(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
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

const openDatasetStep = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
  await user.click(screen.getByRole('button', { name: 'Next' }));
  await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
  await user.click(screen.getByRole('button', { name: 'Next' }));
};

const pickFilesetFile = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
  await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
  await user.click(await screen.findByRole('option', { name: 'generated' }));
  await user.click(await screen.findByRole('combobox', { name: 'File' }));
  await user.click(await screen.findByRole('option', { name: 'output/part-0.parquet' }));
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
  vi.mocked(filesDownloadFile).mockResolvedValue(parquetFile(PARQUET.twoRows));
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

    await openDatasetStep(user);
    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await pickFilesetFile(user);
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    const [, spec] = vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0];
    expect(spec.spec).toMatchObject({ dataset: expect.stringMatching(/#dataset\.parquet$/) });
    const datasetUpload = vi
      .mocked(filesUploadFile)
      .mock.calls.find(([, , name]) => name === 'dataset.parquet');
    expect(new Uint8Array(await (datasetUpload?.[3] as File).arrayBuffer())).toEqual(
      new Uint8Array(await parquetFile(PARQUET.twoRows).arrayBuffer())
    );
  });

  it('reads every Parquet batch from the source fileset in place when asked', async () => {
    server.use(
      http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({
          data: ['output/part-0.parquet', 'output/part-1.parquet'].map((path) => ({
            file_ref: `${DEFAULT_WORKSPACE}/generated#${path}`,
            file_url: `/${path}`,
            path,
            size: 10,
          })),
        })
      )
    );
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.click(screen.getByRole('radio', { name: 'Choose from a fileset' }));
    await user.click(await screen.findByRole('combobox', { name: 'Fileset' }));
    await user.click(await screen.findByRole('option', { name: 'generated' }));
    await user.click(await screen.findByRole('combobox', { name: 'File' }));
    await user.click(await screen.findByRole('option', { name: 'All 2 Parquet files in output/' }));
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    const [, spec] = vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0];
    expect(spec.spec).toMatchObject({
      dataset: `${DEFAULT_WORKSPACE}/generated#output/*.parquet`,
    });
    expect(vi.mocked(filesUploadFile).mock.calls.map(([, , name]) => name)).not.toContain(
      'dataset.parquet'
    );
  });

  it('forgets the fileset file after switching back to upload', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await pickFilesetFile(user);
    await user.click(screen.getByRole('radio', { name: 'Upload a file' }));
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    expect(screen.getByLabelText('Add Dataset')).toBeInTheDocument();
    expect(await screen.findByText('Add a dataset')).toBeVisible();
    expect(evaluatorCreateEvaluateJob).not.toHaveBeenCalled();
  });

  it('still shows the picked file after stepping back and forward', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await pickFilesetFile(user);
    await user.click(screen.getByRole('button', { name: 'Back' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByRole('combobox', { name: 'File' })).toHaveTextContent(
      'output/part-0.parquet'
    );
  });

  it('shows why a fileset file could not be downloaded', async () => {
    vi.mocked(filesDownloadFile).mockRejectedValue(new Error('Unable to find base file.'));
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await pickFilesetFile(user);

    expect(await screen.findByText('Unable to find base file.')).toBeVisible();
  });

  it('shows why a fileset file is not a usable dataset', async () => {
    vi.mocked(filesDownloadFile).mockResolvedValue(parquetFile(PARQUET.empty));
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await pickFilesetFile(user);

    expect(await screen.findByText('File contains no data')).toBeVisible();
  });
});

describe('SubmitEvaluationModal uploaded Parquet dataset', () => {
  it('stores an uploaded Parquet file as-is and points the config at it', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.upload(screen.getByLabelText('Add Dataset'), parquetFile(PARQUET.twoRows));
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    const [, spec] = vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0];
    expect(spec.spec).toMatchObject({ dataset: expect.stringMatching(/#dataset\.parquet$/) });
    expect(filesUploadFile).toHaveBeenCalledWith(
      DEFAULT_WORKSPACE,
      expect.any(String),
      'dataset.parquet',
      expect.any(File),
      expect.anything()
    );
  });

  it('stores several uploaded Parquet files as parts of one dataset', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
    await user.upload(screen.getByLabelText('Add Dataset'), [
      parquetFile(PARQUET.twoRows, 'a.parquet'),
      parquetFile(PARQUET.twoRows, 'b.parquet'),
    ]);
    await user.upload(
      screen.getByLabelText('Select Evaluator Config'),
      new File([EVAL_CONFIG], 'eval-config.yaml', { type: 'application/yaml' })
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evaluatorCreateEvaluateJob).toHaveBeenCalledTimes(1));
    const [, spec] = vi.mocked(evaluatorCreateEvaluateJob).mock.calls[0];
    expect(spec.spec).toMatchObject({ dataset: expect.stringMatching(/#dataset\/\*\.parquet$/) });
    expect(
      vi
        .mocked(filesUploadFile)
        .mock.calls.map(([, , name, file]) => [name, (file as File).name])
        .filter(([name]) => name.startsWith('dataset'))
    ).toEqual([
      ['dataset/part-0.parquet', 'a.parquet'],
      ['dataset/part-1.parquet', 'b.parquet'],
    ]);
  });

  it('refuses to combine Parquet with another format', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
    await user.upload(screen.getByLabelText('Add Dataset'), [
      parquetFile(PARQUET.twoRows, 'a.parquet'),
      new File(['prompt,expected\nhi,hi\n'], 'b.csv'),
    ]);

    expect(await screen.findByText('Only Parquet files can be added together.')).toBeVisible();
  });

  it('keeps Submit from sending a fileset dataset that was never picked', async () => {
    const user = userEvent.setup();
    renderModal();

    await openDatasetStep(user);
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
