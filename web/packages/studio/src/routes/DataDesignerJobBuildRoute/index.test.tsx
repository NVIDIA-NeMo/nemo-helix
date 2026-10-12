// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { CreateJobRequest as DataDesignerJobRequest } from '@nemo/sdk/generated/data-designer/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { DataDesignerJobBuildRoute } from '@studio/routes/DataDesignerJobBuildRoute';
import type { DataDesignerGeneratedState } from '@studio/routes/DataDesignerJobBuildRoute/aiSeed';
import { getDataDesignerJobBuildRoute } from '@studio/routes/utils';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { createMemoryRouter, RouterProvider } from 'react-router';

const BUILD_ROUTE = ROUTES.workspace.dataDesignerJobBuild;
const BUILD_PATH = getDataDesignerJobBuildRoute('default');

const generatedJobRequest: DataDesignerJobRequest = {
  name: 'phishing-emails',
  spec: {
    num_records: 400,
    config: {
      columns: [
        {
          name: 'topic',
          column_type: 'sampler',
          sampler_type: 'category',
          params: { values: ['invoice', 'password reset'] },
        },
        {
          name: 'email_body',
          column_type: 'llm-text',
          prompt: 'Write an email about {{ topic }}.',
          model_alias: 'gen',
        },
      ],
      model_configs: [{ alias: 'gen', model: 'default/llama-3.3', provider: 'default/nim' }],
    },
  },
} as unknown as DataDesignerJobRequest;

const PREVIEW_URL = `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/default/preview`;

const respondToPreview = (...frames: unknown[]) =>
  server.use(
    http.post(
      PREVIEW_URL,
      () =>
        new HttpResponse(frames.map((frame) => JSON.stringify(frame)).join('\n'), {
          headers: { 'Content-Type': 'application/x-ndjson' },
        })
    )
  );

const renderBuildRoute = (state?: DataDesignerGeneratedState, search = '') => {
  const router = createMemoryRouter(
    [{ path: BUILD_ROUTE, element: <DataDesignerJobBuildRoute /> }],
    { initialEntries: [{ pathname: BUILD_PATH, search, state }] }
  );
  return render(
    <TestProviders>
      <RouterProvider router={router} />
    </TestProviders>
  );
};

describe('DataDesignerJobBuildRoute', () => {
  it('loads a generated job request from router state into the builder', async () => {
    renderBuildRoute({ generatedJobRequest });

    expect(await screen.findByText('topic')).toBeInTheDocument();
    expect(await screen.findByText('email_body')).toBeInTheDocument();
  });

  it('opens empty when there is no seed in router state', async () => {
    renderBuildRoute();

    expect(await screen.findByText(/No columns yet/i)).toBeInTheDocument();
  });

  it('reports a template model the workspace does not serve', async () => {
    renderBuildRoute(undefined, '?template=sft-instruction');

    expect(
      await screen.findByText(/config references a model your workspace doesn't have/i)
    ).toBeInTheDocument();
  });

  it('shows preview records as table rows and keeps the logs', async () => {
    respondToPreview(
      { kind: 'log', level: 'info', message: 'Preview generation complete' },
      {
        kind: 'dataset',
        records: [
          { email_body: 'Your invoice is overdue.', topic: 'invoice' },
          { email_body: 'Reset your password now.', topic: 'password reset' },
        ],
      },
      { kind: 'done' }
    );
    renderBuildRoute({ generatedJobRequest });
    await screen.findByText('email_body');

    fireEvent.click(screen.getByRole('button', { name: /Preview \d+ rows/ }));

    expect(await screen.findByText('Preview · 2 rows')).toBeInTheDocument();
    const table = screen.getByRole('table');
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((h) => h.textContent)
    ).toEqual(['#', 'topic', 'email_body']);
    expect(within(table).getByText('Reset your password now.')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Preview logs' }));
    expect(await screen.findByText(/Preview generation complete/)).toBeInTheDocument();
  });

  it('shows the preview error when the stream fails', async () => {
    respondToPreview(
      { kind: 'log', level: 'error', message: 'An error occurred: model unavailable' },
      { kind: 'error', message: 'model unavailable' }
    );
    renderBuildRoute({ generatedJobRequest });
    await screen.findByText('email_body');

    fireEvent.click(screen.getByRole('button', { name: /Preview \d+ rows/ }));

    expect(await screen.findByText(/The preview failed: model unavailable/)).toBeInTheDocument();
    expect(screen.getByText('Preview · failed')).toBeInTheDocument();
  });
});
