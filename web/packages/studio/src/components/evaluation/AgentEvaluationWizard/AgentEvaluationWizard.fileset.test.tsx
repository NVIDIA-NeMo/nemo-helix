// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
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
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { createRunEvaluation } from '@studio/components/evaluation/experimentEvalConfig';
import type { DownloadFileAsArrayBufferArgs } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { PARQUET, parquetFile } from '@studio/tests/util/parquetFixtures';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import type { ReactNode } from 'react';

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

vi.mock('@nemo/common/src/components/form/VariableTextArea', async (importOriginal) => {
  const { forwardRef } = await import('react');
  return {
    ...(await importOriginal<typeof import('@nemo/common/src/components/form/VariableTextArea')>()),
    VariableTextArea: forwardRef(function MockVariableTextArea(
      {
        value,
        onChange,
        attributes,
      }: {
        value: string;
        onChange: (value: string) => void;
        attributes?: { TextAreaElement?: { 'aria-label'?: string } };
      },
      ref: React.Ref<HTMLTextAreaElement>
    ) {
      return (
        <textarea
          ref={ref}
          aria-label={attributes?.TextAreaElement?.['aria-label']}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      );
    }),
  };
});

vi.mock('@studio/components/evaluation/JudgeModelSelect', async () => {
  const { useController } = await import('react-hook-form');
  return {
    JudgeModelSelect: ({
      formFieldName,
      slotLabel,
    }: {
      formFieldName: string;
      slotLabel: ReactNode;
    }) => {
      const { field, fieldState } = useController({ name: formFieldName });
      return (
        <>
          <label>
            {slotLabel}
            <input value={field.value} onChange={field.onChange} />
          </label>
          {fieldState.error ? <span>{fieldState.error.message}</span> : null}
        </>
      );
    },
  };
});

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

const DATASET_REF = `${DEFAULT_WORKSPACE}/generated#output/part-0.parquet`;

const configReading = (right: string) => `prompt_template: "{{ item.prompt }}"
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
        right_template: "{{ ${right} }}"
`;

const EVAL_CONFIG = configReading('item.expected');

const CONFIG_FILESET = 'eval-configs';

const CONFIG_FILESET_FILES: Record<string, string> = {
  'eval-config.yaml': EVAL_CONFIG,
  'configs/strict.json': JSON.stringify({ prompt_template: '{{ item.prompt }}' }),
  'logs/eval-config.yaml': EVAL_CONFIG,
  'notes.txt': 'not a config',
};

const filesetFile = (fileset: string, path: string) => ({
  file_ref: `${DEFAULT_WORKSPACE}/${fileset}#${path}`,
  file_url: `/${path}`,
  path,
  size: 10,
});

const mockListApis = (datasetPaths = ['output/part-0.parquet']) => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: [] })
    ),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), () =>
      HttpResponse.json({ data: [] })
    ),
    http.get(mockApiUrl(getFilesListFilesetsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: CONFIG_FILESET, workspace: DEFAULT_WORKSPACE }],
        pagination: { total: 1, page: 1, page_size: 20 },
      })
    ),
    http.get(
      mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'),
      ({ params: { name } }) =>
        HttpResponse.json({
          data:
            name === CONFIG_FILESET
              ? Object.keys(CONFIG_FILESET_FILES).map((path) => filesetFile(CONFIG_FILESET, path))
              : datasetPaths.map((path) => filesetFile('generated', path)),
        })
    )
  );
};

type User = ReturnType<typeof userEvent.setup>;

const openConfigurationStep = async (user: User) => {
  await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
  await user.click(screen.getByRole('button', { name: 'Next' }));
  await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
  await user.click(screen.getByRole('button', { name: 'Next' }));
};

const pickDataset = async (user: User, ref = DATASET_REF) => {
  await user.click(await screen.findByLabelText('Dataset file ref'));
  await user.paste(ref);
};

const chooseConfigKind = async (user: User, name: string) => {
  await user.click(
    within(await screen.findByRole('radiogroup', { name: 'Eval configuration source' })).getByRole(
      'radio',
      { name }
    )
  );
};

const uploadConfig = async (user: User, config = EVAL_CONFIG) => {
  await user.upload(
    screen.getByLabelText('Eval configuration file'),
    new File([config], 'eval-config.yaml', { type: 'application/yaml' })
  );
};

const clickNextWhenReady = async (user: User) => {
  await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
  await user.click(screen.getByRole('button', { name: 'Next' }));
};

/** Next out of Configuration; parallelism is the one field only that step shows. */
const leaveConfiguration = async (user: User) => {
  await clickNextWhenReady(user);
  await waitFor(() =>
    expect(screen.queryByRole('spinbutton', { name: 'Parallel Requests' })).not.toBeInTheDocument()
  );
};

const nameAndRun = async (user: User) => {
  await user.type(await screen.findByLabelText('Evaluation Name'), 'run-1');
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run Evaluation' })).toBeEnabled());
  await user.click(screen.getByRole('button', { name: 'Run Evaluation' }));
};

const submittedSpec = async () => {
  await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
  return vi.mocked(evalsCreateEvaluateJob).mock.calls[0][1].spec;
};

const uploadedPaths = () => vi.mocked(filesUploadFile).mock.calls.map(([, , path]) => path);

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

beforeEach(() => {
  mockListApis();
  vi.mocked(filesDownloadFile).mockImplementation(async (_workspace, fileset, path) =>
    fileset === CONFIG_FILESET
      ? new Blob([CONFIG_FILESET_FILES[path] ?? ''])
      : parquetFile(PARQUET.twoRows)
  );
  vi.mocked(evalsCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(createExperiment).mockResolvedValue({
    id: 'grp_new',
    name: 'model-update-tests',
  } as never);
  vi.mocked(createRunEvaluation).mockResolvedValue('eval_new' as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('AgentEvaluationWizard dataset', () => {
  it('reads the picked dataset in place and stores only the eval config', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await chooseConfigKind(user, 'Upload');
    await uploadConfig(user);
    await leaveConfiguration(user);
    await nameAndRun(user);

    expect(await submittedSpec()).toMatchObject({ dataset: DATASET_REF });
    expect(uploadedPaths()).toEqual(['eval-config.yaml']);
    expect(createRunEvaluation).toHaveBeenCalledWith(
      DEFAULT_WORKSPACE,
      expect.objectContaining({ datasetName: 'generated' })
    );
  });

  it('reads every Parquet shard beside the picked one once the box is ticked', async () => {
    mockListApis(['output/part-0.parquet', 'output/part-1.parquet']);
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await user.click(
      await screen.findByRole('checkbox', { name: 'Use All 2 Parquet Files in output/' })
    );
    await chooseConfigKind(user, 'Upload');
    await uploadConfig(user);
    await leaveConfiguration(user);
    await nameAndRun(user);

    expect(await submittedSpec()).toMatchObject({
      dataset: `${DEFAULT_WORKSPACE}/generated#output/*.parquet`,
    });
  });

  it('reads only the picked shard unless the box is ticked', async () => {
    mockListApis(['output/part-0.parquet', 'output/part-1.parquet']);
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    expect(
      await screen.findByRole('checkbox', { name: 'Use All 2 Parquet Files in output/' })
    ).not.toBeChecked();
    await chooseConfigKind(user, 'Upload');
    await uploadConfig(user);
    await leaveConfiguration(user);
    await nameAndRun(user);

    expect(await submittedSpec()).toMatchObject({ dataset: DATASET_REF });
  });

  it('will not leave Configuration without a dataset', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await user.click(await screen.findByRole('button', { name: 'Next' }));

    expect(await screen.findByText('Select an input file.')).toBeVisible();
    expect(screen.getByRole('spinbutton', { name: 'Parallel Requests' })).toBeInTheDocument();
  });

  it('holds Next and says why while the picked file is still being read', async () => {
    vi.mocked(filesDownloadFile).mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);

    expect(await screen.findByText('Reading the selected file...')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });

  it('shows why the picked file could not be downloaded', async () => {
    vi.mocked(filesDownloadFile).mockRejectedValue(new Error('Unable to find base file.'));
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);

    expect(await screen.findByText('Unable to find base file.')).toBeVisible();
  });

  it('shows why the picked file is not a usable dataset', async () => {
    vi.mocked(filesDownloadFile).mockResolvedValue(parquetFile(PARQUET.empty));
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);

    expect(await screen.findByText('File contains no data')).toBeVisible();
  });
});

const ROW_PROMPT = 'prompt:\n{{ item.prompt }}\n\nexpected:\n{{ item.expected }}';

describe('AgentEvaluationWizard guided eval config', () => {
  it('sends every field as a text prompt and scores the response with one LLM judge', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    expect(await screen.findByLabelText('Agent Prompt')).toHaveValue(ROW_PROMPT);
    await user.type(screen.getByLabelText('Judge Model'), 'default/judge');
    await leaveConfiguration(user);
    await nameAndRun(user);

    const spec = await submittedSpec();
    expect(spec).toMatchObject({
      dataset: DATASET_REF,
      prompt_template: ROW_PROMPT,
      target: { body: { messages: [{ role: 'user', content: '{{ prompt }}' }] } },
      metrics: [
        {
          metric_type: 'llm-judge',
          payload: {
            metric: {
              prompt_template: {
                messages: [
                  {
                    role: 'user',
                    content: `${ROW_PROMPT}\n\nResponse to evaluate:\n{{ sample.output_text }}`,
                  },
                ],
              },
            },
          },
        },
      ],
    });
    expect(spec.metrics).toHaveLength(1);
    expect(spec).not.toHaveProperty('field_mapping');
    expect(uploadedPaths()).toEqual(['eval-config.json']);
  });

  it('sends the agent prompt as edited', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    const prompt = await screen.findByLabelText('Agent Prompt');
    await user.clear(prompt);
    await user.paste('Q: {{ item.prompt }}');
    await user.type(screen.getByLabelText('Judge Model'), 'default/judge');
    await leaveConfiguration(user);
    await nameAndRun(user);

    expect(await submittedSpec()).toMatchObject({ prompt_template: 'Q: {{ item.prompt }}' });
  });

  it('will not leave Configuration without a judge model', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await clickNextWhenReady(user);

    expect(await screen.findByText('Select a judge model.')).toBeVisible();
    expect(screen.getByRole('spinbutton', { name: 'Parallel Requests' })).toBeInTheDocument();
  });
});

const configPicker = () =>
  within(screen.getByRole('group', { name: 'Eval configuration fileset' }));

const pickFilesetConfig = async (user: User, path: string) => {
  await chooseConfigKind(user, 'Choose from Fileset');
  await user.click(await configPicker().findByRole('combobox', { name: 'Fileset' }));
  await user.click(await screen.findByRole('option', { name: CONFIG_FILESET }));
  await user.click(await configPicker().findByRole('combobox', { name: 'File' }));
  await user.click(await screen.findByRole('option', { name: path }));
};

describe('AgentEvaluationWizard existing eval config', () => {
  it("lists the config's metrics read-only and runs its text prompt", async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await chooseConfigKind(user, 'Upload');
    await uploadConfig(user);

    expect(await screen.findByText('string-check')).toBeVisible();
    expect(screen.queryByLabelText('Judge Prompt')).not.toBeInTheDocument();

    await leaveConfiguration(user);
    await nameAndRun(user);
    expect(await submittedSpec()).toMatchObject({
      prompt_template: '{{ item.prompt }}',
      target: { body: { messages: [{ role: 'user', content: '{{ prompt }}' }] } },
    });
  });

  it('blocks a config that reads a field the dataset does not have', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await chooseConfigKind(user, 'Upload');
    await uploadConfig(user, configReading('item.missing'));

    expect(
      await screen.findByText('Reads "missing", which is not a dataset column or a mapped field.')
    ).toBeVisible();
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });

  it('offers only JSON and YAML files outside logs/', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await chooseConfigKind(user, 'Choose from Fileset');
    await user.click(await configPicker().findByRole('combobox', { name: 'Fileset' }));
    await user.click(await screen.findByRole('option', { name: CONFIG_FILESET }));
    await user.click(await configPicker().findByRole('combobox', { name: 'File' }));

    const options = await screen.findAllByRole('option');
    expect(options.map((option) => option.textContent)).toEqual([
      'configs/strict.json',
      'eval-config.yaml',
    ]);
  });

  it('submits a YAML config picked from a fileset like an uploaded one', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await pickFilesetConfig(user, 'eval-config.yaml');
    await leaveConfiguration(user);
    await nameAndRun(user);

    expect(await submittedSpec()).toMatchObject({ prompt_template: '{{ item.prompt }}' });
    expect(uploadedPaths()).toEqual(['eval-config.yaml']);
  });

  it('shows why a fileset config is not a usable eval config', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await pickFilesetConfig(user, 'configs/strict.json');

    expect(await screen.findByText(/must contain a non-empty "metrics" array/)).toBeVisible();
  });

  it('forgets the fileset config after switching back to upload', async () => {
    const user = userEvent.setup();
    renderWizard();

    await openConfigurationStep(user);
    await pickDataset(user);
    await pickFilesetConfig(user, 'eval-config.yaml');
    await chooseConfigKind(user, 'Upload');
    await clickNextWhenReady(user);

    expect(screen.getByLabelText('Eval configuration file')).toBeInTheDocument();
    expect(await screen.findByText('Select an eval configuration')).toBeVisible();
    expect(evalsCreateEvaluateJob).not.toHaveBeenCalled();
  });
});
