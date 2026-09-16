// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  PlatformJobResponse,
  PlatformJobResponsesPage,
  PlatformJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { JobsDataView } from '@studio/components/dataViews/JobsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { getWorkspaceJobsRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { Link } from 'react-router';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'false');
});

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;
const WORKSPACE = workspace1.workspace;

const makeJob = (name: string): PlatformJobResponse => ({
  id: `id-${name}`,
  attempt_id: `attempt-${name}`,
  name,
  workspace: WORKSPACE,
  source: 'evaluator-metrics',
  fileset: 'fileset-1',
  status: PlatformJobStatus.completed,
  platform_spec: { steps: [] },
  created_at: '2025-06-01T10:00:00Z',
  updated_at: '2025-06-01T12:00:00Z',
});

const page = (name: string): PlatformJobResponsesPage => ({
  data: [makeJob(name)],
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: 1,
    total_pages: 1,
    total_results: 1,
  },
});

const renderComponent = () =>
  renderRoute(<JobsDataView />, {
    history: getWorkspaceJobsRoute(WORKSPACE),
    routes: [{ path: ROUTES.workspace.jobs, element: <JobsDataView /> }],
  });

const isSearch = (request: Request) => new URL(request.url).search.includes('needle');

describe('JobsDataView loading state', () => {
  it('drops the current rows while a changed query is in flight', async () => {
    let release: (() => void) | undefined;
    server.use(
      http.get(JOBS_URL, async ({ request }) => {
        if (isSearch(request)) {
          await new Promise<void>((resolve) => {
            release = resolve;
          });
          return HttpResponse.json(page('searched-job'));
        }
        return HttpResponse.json(page('listed-job'));
      })
    );

    renderComponent();
    expect(await screen.findByText('listed-job')).toBeInTheDocument();

    await userEvent.type(screen.getByPlaceholderText('Search by name'), 'needle');

    await waitFor(() => expect(screen.queryByText('listed-job')).not.toBeInTheDocument());

    release?.();
    expect(await screen.findByText('searched-job')).toBeInTheDocument();
  });

  it('keeps cached rows on screen while an unchanged query refetches', async () => {
    let searchSeen = false;
    let refetches = 0;
    let hold: (() => void) | undefined;
    server.use(
      http.get(JOBS_URL, async ({ request }) => {
        if (isSearch(request)) {
          searchSeen = true;
          return HttpResponse.json(page('searched-job'));
        }
        if (searchSeen) {
          refetches += 1;
          await new Promise<void>((resolve) => {
            hold = resolve;
          });
        }
        return HttpResponse.json(page('listed-job'));
      })
    );

    renderComponent();
    const search = screen.getByPlaceholderText('Search by name');
    expect(await screen.findByText('listed-job')).toBeInTheDocument();

    await userEvent.type(search, 'needle');
    expect(await screen.findByText('searched-job')).toBeInTheDocument();

    await userEvent.clear(search);

    await waitFor(() => expect(refetches).toBe(1));
    expect(screen.getByText('listed-job')).toBeInTheDocument();

    hold?.();
  });

  it('does not re-skeleton when navigating away and back', async () => {
    let requests = 0;
    server.use(
      http.get(JOBS_URL, () => {
        requests += 1;
        return HttpResponse.json(page('listed-job'));
      })
    );

    renderRoute(<JobsDataView />, {
      history: getWorkspaceJobsRoute(WORKSPACE),
      routes: [
        {
          path: ROUTES.workspace.jobs,
          element: (
            <>
              <Link to="/elsewhere">leave</Link>
              <JobsDataView />
            </>
          ),
        },
        { path: '/elsewhere', element: <Link to={getWorkspaceJobsRoute(WORKSPACE)}>back</Link> },
      ],
    });

    expect(await screen.findByText('listed-job')).toBeInTheDocument();
    const firstCount = requests;

    await userEvent.click(screen.getByRole('link', { name: 'leave' }));
    expect(await screen.findByRole('link', { name: 'back' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('link', { name: 'back' }));

    // Cached rows must be on screen on the very first paint after returning.
    expect(screen.getByText('listed-job')).toBeInTheDocument();
    expect(requests).toBeGreaterThanOrEqual(firstCount);
  });
});
