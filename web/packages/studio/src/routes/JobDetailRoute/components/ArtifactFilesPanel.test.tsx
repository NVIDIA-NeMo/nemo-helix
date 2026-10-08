// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { HelixJobResultResponse } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { ArtifactFilesPanel } from '@studio/routes/JobDetailRoute/components/ArtifactFilesPanel';
import { render, screen } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const FILES_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/artifacts/files`;
const result: HelixJobResultResponse = {
  name: 'routing',
  job: 'optimization',
  workspace: DEFAULT_WORKSPACE,
  artifact_storage_type: 'fileset',
  artifact_url: `fileset://${DEFAULT_WORKSPACE}/artifacts#results/routing`,
};

const renderPanel = () =>
  render(
    <ArtifactFilesPanel
      workspace={DEFAULT_WORKSPACE}
      results={[result]}
      isLoading={false}
      jobStatus="completed"
    />
  );

describe('ArtifactFilesPanel', () => {
  it('shows loading while files are being fetched, then displays the artifact row', async () => {
    let releaseFiles = () => {};
    const filesReady = new Promise<void>((resolve) => {
      releaseFiles = resolve;
    });
    server.use(
      http.get(FILES_URL, async () => {
        await filesReady;
        return HttpResponse.json({
          data: [{ path: 'results/routing/agent.yaml', size: 128, file_ref: 'agent-config' }],
        });
      })
    );

    renderPanel();
    try {
      expect(screen.getByLabelText('Loading artifact routing...')).toBeInTheDocument();
    } finally {
      releaseFiles();
    }
    expect(
      await screen.findByRole('button', { name: /results\/routing\/agent.yaml/ })
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('Loading artifact routing...')).not.toBeInTheDocument();
  });

  it.each([
    { scenario: 'empty directory', files: [] },
    {
      scenario: 'unrelated files',
      files: [{ path: 'results/other/agent.yaml', size: 128, file_ref: 'other-config' }],
    },
  ])('shows an empty state when the file listing contains $scenario', async ({ files }) => {
    server.use(http.get(FILES_URL, () => HttpResponse.json({ data: files })));

    renderPanel();

    expect(await screen.findByText(/No files found for artifact/)).toHaveTextContent('routing');
    expect(screen.queryByLabelText('Loading artifact routing...')).not.toBeInTheDocument();
  });

  it('shows an error when artifact files cannot be loaded', async () => {
    server.use(
      http.get(FILES_URL, () => HttpResponse.json({ detail: 'Missing fileset' }, { status: 404 }))
    );

    renderPanel();

    expect(await screen.findByText(/Could not load artifact/)).toHaveTextContent('routing');
    expect(screen.queryByLabelText('Loading artifact routing...')).not.toBeInTheDocument();
  });
});
