// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import {
  type HelixJobResponse,
  type HelixJobResponsesPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { IN_PROGRESS_JOB_STATUSES } from '@studio/constants/platformJobs';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { getNewCustomizationJobRoute } from '@studio/routes/utils';
import { RecentJobsTable } from '@studio/routes/WorkspaceDashboardHomeRoute/RecentJobsTable';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { generatePath } from 'react-router';

const mockEnvironment = vi.hoisted(() => ({ customizerEnabled: true }));

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get CUSTOMIZER_ENABLED() {
    return mockEnvironment.customizerEnabled;
  },
}));

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;
const WORKSPACE = workspace1.workspace;
// "Recent" uses the real clock; fixtures count back from one instant for exact gaps.
const NOW = Date.now();
const hoursAgo = (hours: number) => new Date(NOW - hours * 60 * 60 * 1000).toISOString();

const makeJob = (overrides: Partial<HelixJobResponse> = {}): HelixJobResponse => ({
  id: 'job-id-1',
  attempt_id: 'attempt-1',
  name: 'my-job',
  workspace: WORKSPACE,
  source: 'evaluator-metrics',
  fileset: 'fileset-1',
  status: HelixJobStatus.active,
  platform_spec: { steps: [] },
  created_at: hoursAgo(3),
  updated_at: hoursAgo(1),
  ...overrides,
});

const makeJobsPage = (jobs: HelixJobResponse[]): HelixJobResponsesPage => ({
  data: jobs,
  pagination: {
    page: 1,
    page_size: 10,
    current_page_size: jobs.length,
    total_pages: 1,
    total_results: jobs.length,
  },
});

const renderComponent = (search = '') =>
  renderRoute(<RecentJobsTable />, {
    history: `${generatePath(ROUTES.workspace.dashboard, { workspace: WORKSPACE })}${search}`,
    routes: [
      { path: ROUTES.workspace.dashboard, element: <RecentJobsTable /> },
      { path: ROUTES.workspace.newCustomizationJob, element: <LocationDisplay /> },
    ],
  });

describe('RecentJobsTable', () => {
  beforeEach(() => {
    mockEnvironment.customizerEnabled = true;
  });

  it('renders the Name, Source, Status, and Elapsed time columns with job rows', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          makeJobsPage([
            makeJob({ name: 'dd-run', source: 'data-designer', status: HelixJobStatus.active }),
            makeJob({ id: 'job-id-2', name: 'eval-run', status: HelixJobStatus.completed }),
          ])
        )
      )
    );

    renderComponent();

    expect(await screen.findByText('dd-run')).toBeInTheDocument();
    expect(screen.getByText('eval-run')).toBeInTheDocument();
    expect(screen.getAllByRole('columnheader').map((header) => header.textContent?.trim())).toEqual(
      ['Name', 'Source', 'Status', 'Elapsed time']
    );
    expect(screen.getByText('Data Designer')).toBeInTheDocument();
    expect(screen.getByText('Evaluator')).toBeInTheDocument();
  });

  it('shows the time from created_at to updated_at once a job finishes or pauses', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          makeJobsPage([
            makeJob({ id: 'job-done', name: 'done-run', status: HelixJobStatus.completed }),
            makeJob({
              id: 'job-paused',
              name: 'paused-run',
              status: HelixJobStatus.paused,
              updated_at: hoursAgo(2.25),
            }),
          ])
        )
      )
    );

    renderComponent();

    const doneRow = await screen.findByRole('row', { name: /done-run/ });
    expect(within(doneRow).getByText('2h')).toBeInTheDocument();
    const pausedRow = screen.getByRole('row', { name: /paused-run/ });
    expect(within(pausedRow).getByText('45m')).toBeInTheDocument();
  });

  it('counts a running job up live from created_at rather than to updated_at', async () => {
    server.use(
      http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([makeJob({ name: 'running-run' })])))
    );

    renderComponent();

    const runningRow = await screen.findByRole('row', { name: /running-run/ });
    expect(within(runningRow).queryByText('2h')).not.toBeInTheDocument();
    expect(within(runningRow).getByText(/^3h( \d+m)?( \d+s)?$/)).toBeInTheDocument();
  });

  it('requests in-progress jobs and jobs that finished in the last 48 hours', async () => {
    let filterParam: string | null = null;
    let sortParam: string | null = null;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        filterParam = params.get('filter');
        sortParam = params.get('sort');
        return HttpResponse.json(makeJobsPage([makeJob()]));
      })
    );

    const before = Date.now();
    renderComponent();
    await screen.findByText('my-job');

    expect(sortParam).toBe('-created_at');
    expect(filterParam).not.toBeNull();
    const filter = JSON.parse(filterParam ?? '');
    const [sourceClause, statusClause] = filter.$and;
    expect(sourceClause.source.$nin).toContain('models-system');

    const [inProgress, recentlyFinished] = statusClause.$or;
    expect(inProgress).toEqual({ status: { $in: IN_PROGRESS_JOB_STATUSES } });
    expect(recentlyFinished.$and[0]).toEqual({ status: { $in: HelixJobTerminalStatuses } });

    const cutoff = Date.parse(recentlyFinished.$and[1].updated_at.$gte);
    const fortyEightHours = 48 * 60 * 60 * 1000;
    expect(cutoff).toBeLessThanOrEqual(before - fortyEightHours + 1000);
    expect(cutoff).toBeGreaterThan(before - fortyEightHours - 60_000);
  });

  it('fetches recent jobs in one large page and pages them in the table', async () => {
    const requests: URLSearchParams[] = [];
    const jobs = Array.from({ length: 12 }, (_, i) =>
      makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
    );
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        requests.push(new URL(request.url).searchParams);
        const page = makeJobsPage(jobs);
        // The API's total counts every job, not just those passing the status filter.
        return HttpResponse.json({
          ...page,
          pagination: { ...page.pagination, total_results: 500 },
        });
      })
    );

    renderComponent('?page=2');

    expect(await screen.findByText('run-11')).toBeInTheDocument();
    expect(screen.getByText('run-12')).toBeInTheDocument();
    expect(screen.queryByText('run-10')).not.toBeInTheDocument();
    expect(screen.getByText('11-12 of 12 items')).toBeInTheDocument();
    expect(requests[0].get('page')).toBe('1');
    expect(requests[0].get('page_size')).toBe('100');
    expect(requests.every((params) => params.get('page') === '1')).toBe(true);
  });

  it('drops finished jobs once they age out of the 48-hour window', async () => {
    server.use(
      http.get(JOBS_URL, () =>
        HttpResponse.json(
          makeJobsPage([
            makeJob({ id: 'job-recent', name: 'recent-run', status: HelixJobStatus.error }),
            makeJob({
              id: 'job-old',
              name: 'old-run',
              status: HelixJobStatus.completed,
              created_at: hoursAgo(50),
              updated_at: hoursAgo(49),
            }),
            makeJob({
              id: 'job-long',
              name: 'long-run',
              created_at: hoursAgo(100),
              updated_at: hoursAgo(60),
            }),
          ])
        )
      )
    );

    renderComponent();

    expect(await screen.findByText('recent-run')).toBeInTheDocument();
    expect(screen.getByText('long-run')).toBeInTheDocument();
    expect(screen.queryByText('old-run')).not.toBeInTheDocument();
    expect(screen.getByText('1-2 of 2 items')).toBeInTheDocument();
  });

  it('steps back to the last page when jobs age out from under the page in view', async () => {
    const recent = Array.from({ length: 10 }, (_, i) =>
      makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
    );
    const agedOut = makeJob({
      id: 'job-old',
      name: 'old-run',
      status: HelixJobStatus.completed,
      created_at: hoursAgo(50),
      updated_at: hoursAgo(49),
    });
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([...recent, agedOut]))));

    renderComponent('?page=2');

    expect(await screen.findByText('1-10 of 10 items')).toBeInTheDocument();
    expect(screen.getByText('run-1')).toBeInTheDocument();
    expect(screen.queryByText(ENTITY_EMPTY_STATES.recentJobs.heading)).not.toBeInTheDocument();
  });

  it('points the empty state at the fine-tuning entry point when there are no recent jobs', async () => {
    const user = userEvent.setup();
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([]))));

    renderComponent();

    expect(await screen.findByText(ENTITY_EMPTY_STATES.recentJobs.heading)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Fine-tune a Model' }));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getNewCustomizationJobRoute(WORKSPACE)
    );
  });

  it('leaves the fine-tuning button out of the empty state when fine-tuning is disabled', async () => {
    mockEnvironment.customizerEnabled = false;
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([]))));

    renderComponent();

    expect(await screen.findByText(ENTITY_EMPTY_STATES.recentJobs.heading)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Fine-tune a Model' })).not.toBeInTheDocument();
  });

  it('shows an error panel when the request fails', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json({ detail: 'boom' }, { status: 500 })));

    renderComponent();

    expect(await screen.findByText(/boom/)).toBeInTheDocument();
  });
});
