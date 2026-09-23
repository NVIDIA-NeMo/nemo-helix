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
const OPTIMIZE_JOBS_URL = `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/jobs/optimize`;

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
    ],
  });

const pickBundle = async (files: File[]) => {
  const dialog = await screen.findByRole('dialog');
  fireEvent.change(within(dialog).getByTestId('optimize-bundle-input'), { target: { files } });
  return dialog;
};

const startButton = (dialog: HTMLElement) =>
  within(dialog).getByRole('button', { name: 'Start study' });

describe('LaunchOptimizeModal', () => {
  it('stages the bundle and submits a study for the agent', async () => {
    const { uploaded, submitted } = mockHelix();
    renderModal();

    const dialog = await pickBundle([
      makeFile('bundle/optimize.yaml', OVERLAY),
      makeFile('bundle/dataset.json', '[]'),
      makeFile('bundle/.DS_Store', ''),
    ]);
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(await screen.findByText('Study detail page')).toBeInTheDocument();
    expect(uploaded.sort()).toEqual(['dataset.json', 'optimize.yaml']);
    expect(submitted).toHaveLength(1);
    expect(submitted[0]?.spec).toMatchObject({
      optimize_config: 'optimize.yaml',
      optimize_config_fileset: expect.stringMatching(new RegExp(`^${workspace}/hermes-optimize-`)),
      agent: agentName,
      workspace,
    });
  });

  it('blocks submit and lists preflight problems', async () => {
    mockHelix();
    renderModal();

    const dialog = await pickBundle([makeFile('bundle/optimize.yaml', OVERLAY)]);

    const problems = await within(dialog).findByTestId('optimize-bundle-problems');
    expect(problems).toHaveTextContent('eval.general.dataset points at "dataset.json"');
    expect(startButton(dialog)).toBeDisabled();
  });

  it('asks which config to run when the bundle holds several', async () => {
    const user = userEvent.setup();
    const { submitted } = mockHelix();
    renderModal();

    const dialog = await pickBundle([
      makeFile('bundle/optimize-a.yaml', OVERLAY),
      makeFile('bundle/optimize-b.yml', OVERLAY),
      makeFile('bundle/agent.yaml', 'name: not-an-optimize-config\n'),
      makeFile('bundle/dataset.json', '[]'),
    ]);

    await user.click(await within(dialog).findByRole('combobox', { name: 'Optimize config' }));
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

  it('rejects a selection with no optimize config', async () => {
    mockHelix();
    renderModal();

    const dialog = await pickBundle([makeFile('bundle/agent.yaml', 'name: calc\n')]);

    expect(
      await within(dialog).findByText(/No optimize config in that selection/)
    ).toBeInTheDocument();
    expect(
      within(dialog).queryByRole('combobox', { name: 'Optimize config' })
    ).not.toBeInTheDocument();
  });

  it('shows the submit error and stays open', async () => {
    mockHelix();
    server.use(
      http.post(OPTIMIZE_JOBS_URL, () =>
        HttpResponse.json({ detail: 'Profile default is not configured' }, { status: 422 })
      )
    );
    renderModal();

    const dialog = await pickBundle([
      makeFile('bundle/optimize.yaml', OVERLAY),
      makeFile('bundle/dataset.json', '[]'),
    ]);
    await waitFor(() => expect(startButton(dialog)).toBeEnabled());
    fireEvent.click(startButton(dialog));

    expect(
      await within(dialog).findByText(/Profile default is not configured/)
    ).toBeInTheDocument();
    expect(screen.queryByText('Study detail page')).not.toBeInTheDocument();
  });
});
