// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { InProgressCustomizationsTable } from '@studio/components/dataViews/InProgressCustomizationsTable';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { renderRoute } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;

const activeJob = {
  id: 'j1',
  name: 'grumpy-purple-gorilla',
  workspace: 'ws',
  source: 'customization',
  status: 'active',
  created_at: '2026-10-13T00:00:00Z',
  spec: {
    model: 'default/qwen3-0.6b',
    output: { name: 'grumpy-purple-gorilla' },
    training: { finetuning_type: 'lora' },
  },
};

const jobsResponse = (jobs: unknown[]) => ({
  data: jobs,
  pagination: { page: 1, page_size: 50, total_results: jobs.length, total_pages: 1 },
});

const renderTable = () =>
  renderRoute(<InProgressCustomizationsTable workspace="ws" />, {
    history: '/workspaces/ws/fine-tune',
    routes: [
      {
        path: ROUTES.workspace.customizationJobList,
        element: <InProgressCustomizationsTable workspace="ws" />,
      },
      {
        path: ROUTES.workspace.customizationJobDetails,
        element: <div>job detail page</div>,
      },
    ],
  });

describe('InProgressCustomizationsTable', () => {
  it('lists a run that has not registered its output yet', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([activeJob]))));

    renderTable();

    expect(await screen.findByText('In Progress')).toBeInTheDocument();
    expect(await screen.findByText('grumpy-purple-gorilla')).toBeInTheDocument();
    expect(screen.getByText('default/qwen3-0.6b')).toBeInTheDocument();
    expect(screen.getByText('LoRA')).toBeInTheDocument();
  });

  it('renders nothing when no run is in flight', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([]))));

    renderTable();

    // Absent rather than an empty table: a permanent empty panel above the catalogue
    // would be noise on every visit.
    await waitFor(() => expect(screen.queryByText('In Progress')).not.toBeInTheDocument());
  });

  it('renders nothing when the jobs lookup fails', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.error()));

    renderTable();

    await waitFor(() => expect(screen.queryByText('In Progress')).not.toBeInTheDocument());
  });

  it('opens the job that will produce the output when a row is clicked', async () => {
    const user = userEvent.setup();
    server.use(http.get(JOBS_URL, () => HttpResponse.json(jobsResponse([activeJob]))));

    renderTable();

    await user.click(await screen.findByText('grumpy-purple-gorilla'));

    expect(await screen.findByText('job detail page')).toBeInTheDocument();
  });
});
