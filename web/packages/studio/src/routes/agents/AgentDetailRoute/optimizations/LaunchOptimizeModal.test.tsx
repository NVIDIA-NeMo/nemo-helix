// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { LaunchOptimizeModal } from '@studio/routes/agents/AgentDetailRoute/optimizations/LaunchOptimizeModal';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const workspace = workspace1.workspace;
const agentName = 'hermes';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const UPLOAD_URL = `${FILESETS_URL}/:name/-/*`;
const OPTIMIZE_JOBS_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy`;

const OVERLAY = `optimizer:
  numeric:
    enabled: true
  search_space:
    temperature:
      type: fabric
      path: models.default.temperature
      values: [0.0, 0.2]
eval:
  general:
    dataset:
      file_path: dataset.json
`;

const makeFile = (relativePath: string, contents: string): File => {
  const file = new File([contents], relativePath.split('/').pop() ?? relativePath, {
    type: 'text/plain',
  });
  Object.defineProperty(file, 'webkitRelativePath', { value: relativePath });
  return file;
};

interface SubmittedStudy {
  spec?: Record<string, unknown>;
}

const mockHelix = () => {
  const uploaded: string[] = [];
  const submitted: SubmittedStudy[] = [];
  server.use(
    http.get(FILESETS_URL, () =>
      HttpResponse.json({
        data: [{ id: 'fs-1', name: 'overrides', workspace }],
        pagination: { total_results: 1 },
      })
    ),
    http.get(`${FILESETS_URL}/:name/files`, () =>
      HttpResponse.json({
        data: [
          { path: 'agent.yaml', size: 10 },
          { path: 'notes.md', size: 10 },
        ],
      })
    ),
    http.post(FILESETS_URL, async ({ request }) => HttpResponse.json(await request.json())),
    http.put(UPLOAD_URL, ({ request }) => {
      uploaded.push(decodeURIComponent(new URL(request.url).pathname.split('/-/')[1] ?? ''));
      return HttpResponse.json({ path: 'ok' });
    }),
    http.post(OPTIMIZE_JOBS_URL, async ({ request }) => {
      const body = (await request.json()) as SubmittedStudy;
      submitted.push(body);
      return HttpResponse.json({ ...body, name: 'study-1', workspace, status: 'created' });
    })
  );
  return { uploaded, submitted };
};

const renderModal = () =>
  renderRoute(undefined, {
    history: getAgentDetailRoute(workspace, agentName),
    routes: [
      {
        path: ROUTES.workspace.agentDetail,
        element: (
          <LaunchOptimizeModal open onClose={vi.fn()} workspace={workspace} agentName={agentName} />
        ),
      },
      { path: ROUTES.workspace.agentOptimizationDetail, element: <div>Study detail page</div> },
      { path: ROUTES.workspace.jobDetail, element: <div>Job detail page</div> },
    ],
  });

const pickStrategy = async (name: string) => {
  const dialog = await screen.findByRole('dialog');
  fireEvent.click(await within(dialog).findByText(name));
  return dialog;
};

const pickSource = async (user: ReturnType<typeof userEvent.setup>, option: string) => {
  await user.click(screen.getByRole('combobox', { name: 'Configuration source' }));
  await user.click(screen.getByRole('option', { name: option }));
};

const pickBundle = async (dialog: HTMLElement, files: File[]) =>
  fireEvent.change(within(dialog).getByTestId('optimize-bundle-input'), { target: { files } });

const startButton = (dialog: HTMLElement) =>
  within(dialog).getByRole('button', { name: 'Run strategy' });

describe('LaunchOptimizeModal', () => {
  it('lists every installed strategy and blocks submit until one is chosen', async () => {
    mockHelix();
    renderModal();

    const dialog = await screen.findByRole('dialog');
    expect(await within(dialog).findByText('LEGACY')).toBeInTheDocument();
    expect(within(dialog).getByText('PROMPT-MASTER')).toBeInTheDocument();
    expect(within(dialog).getByText('Hyperparameter and GA prompt optimization.')).toBeVisible();
    expect(startButton(dialog)).toBeDisabled();
  });

  it('submits a strategy with only the agent', async () => {
    const { submitted, uploaded } = mockHelix();
    renderModal();

    const dialog = await pickStrategy('PROMPT-MASTER');
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(await screen.findByText('Job detail page')).toBeInTheDocument();
    expect(uploaded).toEqual([]);
    expect(submitted[0]?.spec).toEqual({ strategy: 'prompt-master', agent: agentName });
  });

  it('stages an uploaded bundle and submits it for the chosen strategy', async () => {
    const user = userEvent.setup();
    const { uploaded, submitted } = mockHelix();
    renderModal();

    const dialog = await pickStrategy('LEGACY');
    await pickSource(user, 'Upload files');
    await pickBundle(dialog, [
      makeFile('bundle/optimize.yaml', OVERLAY),
      makeFile('bundle/dataset.json', '[]'),
      makeFile('bundle/.DS_Store', ''),
    ]);
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(await screen.findByText('Study detail page')).toBeInTheDocument();
    expect(uploaded.sort()).toEqual(['dataset.json', 'optimize.yaml']);
    expect(submitted[0]?.spec).toMatchObject({
      strategy: 'legacy',
      optimize_config: 'optimize.yaml',
      optimize_config_fileset: expect.stringMatching(new RegExp(`^${workspace}/hermes-optimize-`)),
      agent: agentName,
    });
  });

  it('submits a config from an existing fileset', async () => {
    const user = userEvent.setup();
    const { submitted, uploaded } = mockHelix();
    renderModal();

    const dialog = await pickStrategy('PROMPT-MASTER');
    await pickSource(user, 'overrides');
    await user.click(await screen.findByRole('combobox', { name: 'Config file' }));
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
      'agent.yaml',
    ]);
    await user.click(screen.getByRole('option', { name: 'agent.yaml' }));
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(await screen.findByText('Job detail page')).toBeInTheDocument();
    expect(uploaded).toEqual([]);
    expect(submitted[0]?.spec).toEqual({
      strategy: 'prompt-master',
      agent: agentName,
      optimize_config: 'agent.yaml',
      optimize_config_fileset: `${workspace}/overrides`,
    });
  });

  it('blocks submit and lists preflight problems for a legacy bundle', async () => {
    const user = userEvent.setup();
    mockHelix();
    renderModal();

    const dialog = await pickStrategy('LEGACY');
    await pickSource(user, 'Upload files');
    await pickBundle(dialog, [makeFile('bundle/optimize.yaml', OVERLAY)]);

    const problems = await within(dialog).findByTestId('optimize-bundle-problems');
    expect(problems).toHaveTextContent('eval.general.dataset points at "dataset.json"');
    expect(startButton(dialog)).toBeDisabled();
  });

  it('asks which config to run when the bundle holds several YAML files', async () => {
    const user = userEvent.setup();
    const { submitted } = mockHelix();
    renderModal();

    const dialog = await pickStrategy('LEGACY');
    await pickSource(user, 'Upload files');
    await pickBundle(dialog, [
      makeFile('bundle/optimize-a.yaml', OVERLAY),
      makeFile('bundle/optimize-b.yml', OVERLAY),
      makeFile('bundle/dataset.json', '[]'),
    ]);

    await user.click(await within(dialog).findByRole('combobox', { name: 'Config file' }));
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
      'optimize-a.yaml',
      'optimize-b.yml',
    ]);
    expect(startButton(dialog)).toBeDisabled();

    await user.click(screen.getByRole('option', { name: 'optimize-b.yml' }));
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    await user.click(startButton(dialog));

    await waitFor(() => expect(submitted[0]?.spec?.optimize_config).toBe('optimize-b.yml'));
  });

  it('rejects an upload with no YAML', async () => {
    const user = userEvent.setup();
    mockHelix();
    renderModal();

    const dialog = await pickStrategy('LEGACY');
    await pickSource(user, 'Upload files');
    await pickBundle(dialog, [makeFile('bundle/dataset.json', '[]')]);

    expect(await within(dialog).findByText(/No config in that selection/)).toBeInTheDocument();
    expect(startButton(dialog)).toBeDisabled();
  });

  it('shows the strategy validation error and stays open', async () => {
    mockHelix();
    server.use(
      http.post(OPTIMIZE_JOBS_URL, () =>
        HttpResponse.json(
          { detail: "Spec is not valid for optimization strategy 'strands-harness-optimizer'" },
          { status: 422 }
        )
      )
    );
    renderModal();

    const dialog = await pickStrategy('STRANDS-HARNESS-OPTIMIZER');
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(
      await within(dialog).findByText(/Spec is not valid for optimization strategy/)
    ).toBeInTheDocument();
    expect(screen.queryByText('Job detail page')).not.toBeInTheDocument();
  });
});
