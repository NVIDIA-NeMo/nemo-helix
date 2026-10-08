// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
} from '@nemo/sdk/generated/platform/files';
import type { DownloadFileAsArrayBufferArgs } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { LaunchOptimizeModal } from '@studio/routes/agents/AgentDetailRoute/optimizations/LaunchOptimizeModal';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

/** Contents of the files in the fileset the test picks, keyed by path. */
const filesetContents = new Map<string, string>();

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer:
    () =>
    async ({ path }: DownloadFileAsArrayBufferArgs) =>
      new TextEncoder().encode(filesetContents.get(path) ?? '').buffer,
}));

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

const mockFileset = (files: Record<string, string>) => {
  filesetContents.clear();
  Object.entries(files).forEach(([path, contents]) => filesetContents.set(path, contents));
  server.use(
    http.get(mockApiUrl(getFilesListFilesetsQueryKey, ':workspace'), () =>
      HttpResponse.json({
        data: [{ name: 'curated-bundle', workspace }],
        pagination: { total: 1, page: 1, page_size: 20 },
      })
    ),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({
        data: Object.keys(files).map((path) => ({
          file_ref: `${workspace}/curated-bundle#${path}`,
          file_url: `/${path}`,
          path,
          size: 10,
        })),
      })
    )
  );
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
  fireEvent.change(within(dialog).getByTestId('bundle-input'), { target: { files } });
  return dialog;
};

const pickFileset = async (user: ReturnType<typeof userEvent.setup>) => {
  const dialog = await screen.findByRole('dialog');
  await user.click(within(dialog).getByRole('radio', { name: 'Choose from a fileset' }));
  await user.click(await within(dialog).findByRole('combobox', { name: 'Fileset' }));
  await user.click(await screen.findByRole('option', { name: 'curated-bundle' }));
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
      strategy: 'legacy',
      optimize_config: 'optimize.yaml',
      optimize_config_fileset: expect.stringMatching(new RegExp(`^${workspace}/hermes-optimize-`)),
      agent: agentName,
    });
  });

  it('blocks submit and lists preflight problems', async () => {
    mockHelix();
    renderModal();

    const dialog = await pickBundle([makeFile('bundle/optimize.yaml', OVERLAY)]);

    const problems = await within(dialog).findByTestId('bundle-problems');
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

  describe('from a fileset', () => {
    it('runs the study from the fileset in place', async () => {
      const user = userEvent.setup();
      const { uploaded, submitted } = mockHelix();
      mockFileset({ 'optimize.yaml': OVERLAY, 'dataset.json': '[]' });
      renderModal();

      const dialog = await pickFileset(user);
      await waitFor(() => expect(startButton(dialog)).toBeEnabled());
      await user.click(startButton(dialog));

      expect(await screen.findByText('Study detail page')).toBeInTheDocument();
      expect(uploaded).toEqual([]);
      expect(submitted).toHaveLength(1);
      expect(submitted[0]?.spec).toMatchObject({
        optimize_config: 'optimize.yaml',
        optimize_config_fileset: `${workspace}/curated-bundle`,
        agent: agentName,
      });
      expect(submitted[0]).not.toHaveProperty('custom_fields');
    });

    it('preflights the config against the fileset listing', async () => {
      const user = userEvent.setup();
      mockHelix();
      mockFileset({ 'configs/optimize.yaml': OVERLAY });
      renderModal();

      const dialog = await pickFileset(user);

      const problems = await within(dialog).findByTestId('bundle-problems');
      expect(problems).toHaveTextContent('eval.general.dataset points at "dataset.json"');
      expect(startButton(dialog)).toBeDisabled();
    });

    it('asks which YAML to run when the fileset holds several', async () => {
      const user = userEvent.setup();
      const { submitted } = mockHelix();
      mockFileset({
        'agent.yaml': 'name: not-an-optimize-config\n',
        'optimize.yaml': OVERLAY,
        'dataset.json': '[]',
      });
      renderModal();

      const dialog = await pickFileset(user);
      await user.click(await within(dialog).findByRole('combobox', { name: 'Optimize config' }));
      expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
        'agent.yaml',
        'optimize.yaml',
      ]);
      await user.click(screen.getByRole('option', { name: 'agent.yaml' }));

      expect(
        await within(dialog).findByText(/agent.yaml is not a valid optimize config/)
      ).toBeInTheDocument();
      expect(startButton(dialog)).toBeDisabled();

      await user.click(within(dialog).getByRole('combobox', { name: 'Optimize config' }));
      await user.click(screen.getByRole('option', { name: 'optimize.yaml' }));
      await waitFor(() => expect(startButton(dialog)).toBeEnabled());
      await user.click(startButton(dialog));

      await waitFor(() => expect(submitted[0]?.spec?.optimize_config).toBe('optimize.yaml'));
    });

    it('says when the fileset holds no YAML', async () => {
      const user = userEvent.setup();
      mockHelix();
      mockFileset({ 'dataset.json': '[]' });
      renderModal();

      const dialog = await pickFileset(user);

      expect(await within(dialog).findByText('This fileset has no YAML files.')).toBeVisible();
      expect(startButton(dialog)).toBeDisabled();
    });
  });
});
