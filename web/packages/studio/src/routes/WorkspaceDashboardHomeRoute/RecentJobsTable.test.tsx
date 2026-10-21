// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import {
  type HelixJobResponse,
  type HelixJobResponsesPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { getNewCustomizationJobRoute } from '@studio/routes/utils';
import { RecentJobsTable } from '@studio/routes/WorkspaceDashboardHomeRoute/RecentJobsTable';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { act, renderRoute, screen, within } from '@studio/tests/util/render';
import { focusManager, onlineManager } from '@tanstack/react-query';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useState } from 'react';
import { generatePath, Link } from 'react-router';

const mockEnvironment = vi.hoisted(() => ({ customizerEnabled: true }));

vi.mock('@studio/constants/environment', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/constants/environment')>()),
  get CUSTOMIZER_ENABLED() {
    return mockEnvironment.customizerEnabled;
  },
}));

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;
const WORKSPACE = workspace1.workspace;
const NOW = Date.now();
const CUTOFF_BUCKET_MS = 5 * 60 * 1000;
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

const makeJobsPage = (
  jobs: HelixJobResponse[],
  { page = 1, pageSize = 10, totalResults = jobs.length } = {}
): HelixJobResponsesPage => ({
  data: jobs,
  pagination: {
    page,
    page_size: pageSize,
    current_page_size: jobs.length,
    total_pages: Math.ceil(totalResults / pageSize),
    total_results: totalResults,
  },
});

const pageJobs = (request: Request, jobs: HelixJobResponse[]) => {
  const params = new URL(request.url).searchParams;
  const page = Number(params.get('page'));
  const pageSize = Number(params.get('page_size'));
  return makeJobsPage(jobs.slice((page - 1) * pageSize, page * pageSize), {
    page,
    pageSize,
    totalResults: jobs.length,
  });
};

const makeRuns = (count: number) =>
  Array.from({ length: count }, (_, i) =>
    makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
  );

const refetchOnFocus = () =>
  act(() => {
    focusManager.setFocused(false);
    focusManager.setFocused(true);
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

  afterEach(() => {
    vi.useRealTimers();
    focusManager.setFocused(undefined);
    onlineManager.setOnline(true);
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

  it('updates cells in place when a poll returns new data instead of remounting them', async () => {
    let requestCount = 0;
    server.use(
      http.get(JOBS_URL, () => {
        requestCount += 1;
        return HttpResponse.json(
          makeJobsPage([makeJob({ name: requestCount === 1 ? 'first-name' : 'second-name' })])
        );
      })
    );

    renderComponent();
    await screen.findByText('first-name');
    const [nameCell] = screen.getAllByTestId('line-clamp');

    await refetchOnFocus();

    expect(await screen.findByText('second-name')).toBeInTheDocument();
    expect(requestCount).toBe(2);
    expect(screen.getAllByTestId('line-clamp')[0]).toBe(nameCell);
    expect(nameCell).toHaveTextContent('second-name');
  });

  it('requests in-progress jobs and jobs that finished in the last 48 hours', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date('2026-01-03T12:34:56.789Z'));
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

    renderComponent();
    await screen.findByText('my-job');

    expect(sortParam).toBe('-created_at');
    expect(filterParam).not.toBeNull();
    const filter = JSON.parse(filterParam ?? '');
    const [sourceClause, statusClause] = filter.$and;
    expect(sourceClause.source.$nin).toContain('models-system');
    expect(sourceClause.source.$nin).not.toContain('customization');

    expect(statusClause.$or).toEqual([
      { status: { $nin: HelixJobTerminalStatuses } },
      { updated_at: { $gte: '2026-01-01T12:30:00.000Z' } },
    ]);
  });

  it('moves the 48-hour cutoff forward as polls pass without reloading the page in view', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date('2026-01-03T12:04:40.000Z'));
    const cutoffs: string[] = [];
    let releaseRefetch = () => {};
    server.use(
      http.get(JOBS_URL, async ({ request }) => {
        const filter = JSON.parse(new URL(request.url).searchParams.get('filter') ?? '');
        const cutoff: string = filter.$and[1].$or[1].updated_at.$gte;
        cutoffs.push(cutoff);
        if (cutoffs.length > 1 && cutoff !== cutoffs[0]) {
          await new Promise<void>((resolve) => {
            releaseRefetch = resolve;
          });
          return HttpResponse.json(makeJobsPage([makeJob({ name: 'fresh-run' })]));
        }
        return HttpResponse.json(
          makeJobsPage([makeJob({ name: 'done-run', status: HelixJobStatus.completed })])
        );
      })
    );

    renderComponent();
    await screen.findByText('done-run');
    expect(cutoffs).toEqual(['2026-01-01T12:00:00.000Z']);

    await act(() => vi.advanceTimersByTimeAsync(30_000));
    await vi.waitFor(() => expect(cutoffs.at(-1)).toBe('2026-01-01T12:05:00.000Z'));

    expect(screen.getByText('done-run')).toBeInTheDocument();
    expect(screen.getByText('1-1 of 1 items')).toBeInTheDocument();

    releaseRefetch();
    expect(await screen.findByText('fresh-run')).toBeInTheDocument();
  });

  it('leaves fine-tuning jobs out of the request when fine-tuning is disabled', async () => {
    mockEnvironment.customizerEnabled = false;
    let filterParam: string | null = null;
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        filterParam = new URL(request.url).searchParams.get('filter');
        return HttpResponse.json(makeJobsPage([makeJob()]));
      })
    );

    renderComponent();
    await screen.findByText('my-job');

    const [sourceClause] = JSON.parse(filterParam ?? '').$and;
    expect(sourceClause.source.$nin).toEqual(
      expect.arrayContaining(['models-system', 'customization'])
    );
  });

  it('shows cached rows right away when the table remounts within the cutoff bucket', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.setSystemTime(new Date('2026-01-03T12:10:00.000Z'));
    const filters: (string | null)[] = [];
    let releaseRefetch = () => {};
    server.use(
      http.get(JOBS_URL, async ({ request }) => {
        filters.push(new URL(request.url).searchParams.get('filter'));
        if (filters.length > 1) {
          await new Promise<void>((resolve) => {
            releaseRefetch = resolve;
          });
          return HttpResponse.json(makeJobsPage([makeJob({ name: 'fresh-run' })]));
        }
        return HttpResponse.json(makeJobsPage([makeJob({ name: 'cached-run' })]));
      })
    );
    const RemountableTable = () => {
      const [isShown, setIsShown] = useState(true);
      return (
        <>
          <button type="button" onClick={() => setIsShown((shown) => !shown)}>
            toggle
          </button>
          {isShown && <RecentJobsTable />}
        </>
      );
    };
    renderRoute(<RemountableTable />, {
      history: generatePath(ROUTES.workspace.dashboard, { workspace: WORKSPACE }),
      routes: [{ path: ROUTES.workspace.dashboard, element: <RemountableTable /> }],
    });
    await screen.findByText('cached-run');

    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    await user.click(screen.getByRole('button', { name: 'toggle' }));
    vi.setSystemTime(new Date('2026-01-03T12:14:00.000Z'));
    await user.click(screen.getByRole('button', { name: 'toggle' }));

    expect(screen.getByText('cached-run')).toBeInTheDocument();
    await vi.waitFor(() => expect(filters).toHaveLength(2));
    expect(filters[1]).toBe(filters[0]);

    releaseRefetch();
    expect(await screen.findByText('fresh-run')).toBeInTheDocument();
  });

  it.each(['?sort=name', '?sort=created_at', '?sort=source'])(
    'ignores %s in the URL and requests the newest jobs',
    async (search) => {
      let sortParam: string | null = null;
      server.use(
        http.get(JOBS_URL, ({ request }) => {
          sortParam = new URL(request.url).searchParams.get('sort');
          return HttpResponse.json(makeJobsPage([makeJob()]));
        })
      );

      renderComponent(search);

      expect(await screen.findByText('my-job')).toBeInTheDocument();
      expect(sortParam).toBe('-created_at');
    }
  );

  it('requests the page in view and counts every matching job from the API total', async () => {
    const requests: URLSearchParams[] = [];
    const jobs = Array.from({ length: 12 }, (_, i) =>
      makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
    );
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        requests.push(new URL(request.url).searchParams);
        return HttpResponse.json(pageJobs(request, jobs));
      })
    );

    renderComponent('?page=2');

    expect(await screen.findByText('run-11')).toBeInTheDocument();
    expect(screen.getByText('run-12')).toBeInTheDocument();
    expect(screen.queryByText('run-10')).not.toBeInTheDocument();
    expect(screen.getByText('11-12 of 12 items')).toBeInTheDocument();
    expect(requests.map((params) => [params.get('page'), params.get('page_size')])).toEqual([
      ['2', '10'],
    ]);
  });

  it('requests the next page when the user pages forward', async () => {
    const user = userEvent.setup();
    const jobs = Array.from({ length: 12 }, (_, i) =>
      makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
    );
    server.use(http.get(JOBS_URL, ({ request }) => HttpResponse.json(pageJobs(request, jobs))));

    renderComponent();
    expect(await screen.findByText('1-10 of 12 items')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /next page/i }));

    expect(await screen.findByText('run-11')).toBeInTheDocument();
    expect(screen.queryByText('run-1')).not.toBeInTheDocument();
    expect(screen.getByText('11-12 of 12 items')).toBeInTheDocument();
  });

  it('steps back to the last page when jobs age out from under the page in view', async () => {
    const requestedPages: (string | null)[] = [];
    const recent = Array.from({ length: 10 }, (_, i) =>
      makeJob({ id: `job-${i + 1}`, name: `run-${i + 1}`, status: HelixJobStatus.completed })
    );
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        requestedPages.push(new URL(request.url).searchParams.get('page'));
        return HttpResponse.json(pageJobs(request, recent));
      })
    );

    renderComponent('?page=2');

    expect(await screen.findByText('1-10 of 10 items')).toBeInTheDocument();
    expect(screen.getByText('run-1')).toBeInTheDocument();
    expect(screen.queryByText(ENTITY_EMPTY_STATES.recentJobs.heading)).not.toBeInTheDocument();
    expect(requestedPages).toEqual(['2', '1']);
  });

  it('refetches a cached page that ran past the end before stepping back from it', async () => {
    const user = userEvent.setup();
    let jobs = makeRuns(11);
    server.use(http.get(JOBS_URL, ({ request }) => HttpResponse.json(pageJobs(request, jobs))));

    renderComponent('?page=2');
    expect(await screen.findByText('run-11')).toBeInTheDocument();

    jobs = makeRuns(10);
    await refetchOnFocus();
    expect(await screen.findByText('1-10 of 10 items')).toBeInTheDocument();

    jobs = makeRuns(11);
    await refetchOnFocus();
    expect(await screen.findByText('1-10 of 11 items')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /next page/i }));

    expect(await screen.findByText('run-11')).toBeInTheDocument();
    expect(screen.getByText('11-11 of 11 items')).toBeInTheDocument();
  });

  it('keeps the pager when a page fails to load so the user can page back', async () => {
    const user = userEvent.setup();
    const jobs = makeRuns(12);
    server.use(
      http.get(JOBS_URL, ({ request }) =>
        new URL(request.url).searchParams.get('page') === '2'
          ? HttpResponse.json({ detail: 'boom' }, { status: 403 })
          : HttpResponse.json(pageJobs(request, jobs))
      )
    );

    renderComponent();
    expect(await screen.findByText('1-10 of 12 items')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /next page/i }));
    expect(await screen.findByText(/boom/)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /previous page/i }));
    expect(await screen.findByText('run-1')).toBeInTheDocument();
    expect(screen.queryByText(/boom/)).not.toBeInTheDocument();
  });

  it("doesn't show the previous workspace's jobs while another workspace loads", async () => {
    const user = userEvent.setup();
    const otherWorkspace = 'other-workspace';
    let isOtherRequested = false;
    let releaseOther = () => {};
    server.use(
      http.get(JOBS_URL, async ({ params }) => {
        if (params.workspace !== otherWorkspace) {
          return HttpResponse.json(makeJobsPage([makeJob({ name: 'my-run' })]));
        }
        isOtherRequested = true;
        await new Promise<void>((resolve) => {
          releaseOther = resolve;
        });
        return HttpResponse.json(
          makeJobsPage([makeJob({ name: 'other-run', workspace: otherWorkspace })])
        );
      })
    );
    const DashboardWithSwitcher = () => (
      <>
        <Link to={generatePath(ROUTES.workspace.dashboard, { workspace: otherWorkspace })}>
          switch
        </Link>
        <RecentJobsTable />
      </>
    );
    renderRoute(<DashboardWithSwitcher />, {
      history: generatePath(ROUTES.workspace.dashboard, { workspace: WORKSPACE }),
      routes: [{ path: ROUTES.workspace.dashboard, element: <DashboardWithSwitcher /> }],
    });
    await screen.findByText('my-run');

    await user.click(screen.getByRole('link', { name: 'switch' }));
    await vi.waitFor(() => expect(isOtherRequested).toBe(true));

    expect(screen.queryByText('my-run')).not.toBeInTheDocument();
    releaseOther();
    expect(await screen.findByText('other-run')).toBeInTheDocument();
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

  it('shows the error at once without retrying when the request fails with a client error', async () => {
    let requestCount = 0;
    server.use(
      http.get(JOBS_URL, () => {
        requestCount += 1;
        return HttpResponse.json({ detail: 'boom' }, { status: 403 });
      })
    );

    renderComponent();

    expect(await screen.findByText(/boom/)).toBeInTheDocument();
    expect(requestCount).toBe(1);
  });

  it('retries a server error before showing it', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let requestCount = 0;
    server.use(
      http.get(JOBS_URL, () => {
        requestCount += 1;
        return HttpResponse.json({ detail: 'boom' }, { status: 500 });
      })
    );

    renderComponent();
    await act(() => vi.advanceTimersByTimeAsync(10_000));

    expect(await screen.findByText(/boom/)).toBeInTheDocument();
    expect(requestCount).toBe(DEFAULT_QUERY_RETRY_COUNT + 1);
  });

  it('shows the loading state, not the empty state, while offline before the first load', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([makeJob()]))));
    onlineManager.setOnline(false);

    renderComponent();

    // A header row plus a loading row per item on the page.
    await vi.waitFor(() => expect(screen.getAllByRole('row')).toHaveLength(11));
    expect(screen.queryByText(ENTITY_EMPTY_STATES.recentJobs.heading)).not.toBeInTheDocument();

    act(() => onlineManager.setOnline(true));
    expect(await screen.findByText('my-job')).toBeInTheDocument();
  });

  describe('polling', () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      // Just past a cutoff boundary, so no poll moves the cutoff.
      vi.setSystemTime(Math.floor(NOW / CUTOFF_BUCKET_MS) * CUTOFF_BUCKET_MS + 1000);
    });

    it('polls every 15s while a job is running', async () => {
      let requestCount = 0;
      server.use(
        http.get(JOBS_URL, () => {
          requestCount += 1;
          return HttpResponse.json(makeJobsPage([makeJob({ name: 'running-run' })]));
        })
      );

      renderComponent();
      await screen.findByText('running-run');
      expect(requestCount).toBe(1);

      await act(() => vi.advanceTimersByTimeAsync(14_000));
      expect(requestCount).toBe(1);

      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(requestCount).toBe(2);
    });

    it('polls every 30s when nothing is running', async () => {
      let requestCount = 0;
      server.use(
        http.get(JOBS_URL, () => {
          requestCount += 1;
          return HttpResponse.json(
            makeJobsPage([makeJob({ name: 'done-run', status: HelixJobStatus.completed })])
          );
        })
      );

      renderComponent();
      await screen.findByText('done-run');

      await act(() => vi.advanceTimersByTimeAsync(29_000));
      expect(requestCount).toBe(1);

      await act(() => vi.advanceTimersByTimeAsync(1000));
      expect(requestCount).toBe(2);
    });

    it('keeps the last rows while a poll that failed with a server error is retried', async () => {
      let requestCount = 0;
      server.use(
        http.get(JOBS_URL, () => {
          requestCount += 1;
          if (requestCount === 2) {
            return HttpResponse.json({ detail: 'boom' }, { status: 503 });
          }
          return HttpResponse.json(
            makeJobsPage([makeJob({ name: requestCount === 1 ? 'first-run' : 'second-run' })])
          );
        })
      );

      renderComponent();
      await screen.findByText('first-run');

      await act(() => vi.advanceTimersByTimeAsync(15_000));

      expect(requestCount).toBe(2);
      expect(screen.getByText('first-run')).toBeInTheDocument();
      expect(screen.queryByText(/boom/)).not.toBeInTheDocument();

      await act(() => vi.advanceTimersByTimeAsync(1000));

      expect(await screen.findByText('second-run')).toBeInTheDocument();
      expect(requestCount).toBe(3);
    });

    it('keeps the last rows without retrying a poll that failed with a client error', async () => {
      let requestCount = 0;
      server.use(
        http.get(JOBS_URL, () => {
          requestCount += 1;
          if (requestCount === 2) {
            return HttpResponse.json({ detail: 'boom' }, { status: 403 });
          }
          return HttpResponse.json(
            makeJobsPage([makeJob({ name: requestCount === 1 ? 'first-run' : 'second-run' })])
          );
        })
      );

      renderComponent();
      await screen.findByText('first-run');

      await act(() => vi.advanceTimersByTimeAsync(25_000));

      expect(requestCount).toBe(2);
      expect(screen.getByText('first-run')).toBeInTheDocument();
      expect(screen.queryByText(/boom/)).not.toBeInTheDocument();

      await act(() => vi.advanceTimersByTimeAsync(20_000));

      expect(await screen.findByText('second-run')).toBeInTheDocument();
      expect(requestCount).toBe(3);
    });

    it('keeps showing the error while a failed first load polls again', async () => {
      let requestCount = 0;
      let releaseRetry = () => {};
      server.use(
        http.get(JOBS_URL, async () => {
          requestCount += 1;
          if (requestCount === 1) {
            return HttpResponse.json({ detail: 'boom' }, { status: 403 });
          }
          await new Promise<void>((resolve) => {
            releaseRetry = resolve;
          });
          return HttpResponse.json(makeJobsPage([makeJob({ name: 'recovered-run' })]));
        })
      );

      renderComponent();
      expect(await screen.findByText(/boom/)).toBeInTheDocument();

      await act(() => vi.advanceTimersByTimeAsync(30_000));

      expect(requestCount).toBe(2);
      expect(screen.getByText(/boom/)).toBeInTheDocument();

      releaseRetry();
      expect(await screen.findByText('recovered-run')).toBeInTheDocument();
    });
  });
});
