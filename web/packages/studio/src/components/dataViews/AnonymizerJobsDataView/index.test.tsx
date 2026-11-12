// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { getAnonymizerListRunJobsQueryKey } from '@nemo/sdk/generated/anonymizer/anonymizer';
import type { RunJob, RunJobsPage } from '@nemo/sdk/generated/anonymizer/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/anonymizer/schema/HelixJobStatus';
import { AnonymizerJobsDataView } from '@studio/components/dataViews/AnonymizerJobsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getWorkspaceAnonymizerRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent, { type UserEvent } from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const LIST_URL = mockApiUrl(getAnonymizerListRunJobsQueryKey, ':workspace');
const WORKSPACE = workspace1.workspace;
const FAILURE_DETAIL = 'Audit simulated list failure';

const makeJob = (name: string): RunJob => ({
  id: `${name}-id`,
  name,
  workspace: WORKSPACE,
  status: HelixJobStatus.active,
  spec: {} as RunJob['spec'],
  created_at: '2025-06-01T10:00:00Z',
});

const makePage = (jobs: RunJob[], page = 1, pageSize = 25, totalResults = jobs.length) =>
  ({
    data: jobs,
    pagination: {
      page,
      page_size: pageSize,
      current_page_size: jobs.length,
      total_pages: Math.ceil(totalResults / pageSize),
      total_results: totalResults,
    },
  }) satisfies RunJobsPage;

const failure = () => HttpResponse.json({ detail: FAILURE_DETAIL }, { status: 500 });

const renderComponent = (query = '') =>
  renderRoute(<AnonymizerJobsDataView />, {
    history: `${getWorkspaceAnonymizerRoute(WORKSPACE)}${query}`,
    routes: [{ path: ROUTES.workspace.anonymizer, element: <AnonymizerJobsDataView /> }],
  });

const findErrorPanel = () =>
  screen.findByTestId('error-panel', undefined, { timeout: XL_SELECTOR_TIMEOUT });

describe('AnonymizerJobsDataView', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows the list-load error instead of the first-use state when the request fails', async () => {
    server.use(http.get(LIST_URL, failure));

    renderComponent();

    expect(await findErrorPanel()).toBeInTheDocument();
    expect(screen.getByText(FAILURE_DETAIL)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.queryByTestId('entity-empty-state-first-use')).not.toBeInTheDocument();
    expect(screen.queryByText(ENTITY_EMPTY_STATES.anonymizerJobs.heading)).not.toBeInTheDocument();
  });

  it('shows the error instead of no-results when a filtered search fails', async () => {
    server.use(http.get(LIST_URL, failure));

    renderComponent('?s=foo');

    expect(await findErrorPanel()).toBeInTheDocument();
    expect(screen.queryByTestId('entity-empty-state-no-results')).not.toBeInTheDocument();
  });

  it('retries with the current search, sort and page, and recovers', async () => {
    const user = userEvent.setup();
    const requests: URL[] = [];
    let shouldFail = true;
    server.use(
      http.get(LIST_URL, ({ request }) => {
        requests.push(new URL(request.url));
        return shouldFail ? failure() : HttpResponse.json(makePage([makeJob('foo-2')], 2, 10, 11));
      })
    );

    renderComponent('?s=foo&page=2&page_size=10&sort=created_at');

    await findErrorPanel();
    shouldFail = false;
    await user.click(screen.getByRole('button', { name: 'Retry' }));

    expect(await screen.findByText('foo-2')).toBeInTheDocument();
    expect(screen.queryByTestId('error-panel')).not.toBeInTheDocument();
    const params = requests.at(-1)!.searchParams;
    expect(params.get('filter[name][$like]')).toBe('foo');
    expect(params.get('sort')).toBe('created_at');
    expect(params.get('page')).toBe('2');
    expect(params.get('page_size')).toBe('10');
  });

  it('shows the error when a page change fails', async () => {
    const user = userEvent.setup();
    const firstPage = Array.from({ length: 10 }, (_, i) => makeJob(`job-${i + 1}`));
    server.use(
      http.get(LIST_URL, ({ request }) =>
        new URL(request.url).searchParams.get('page') === '2'
          ? failure()
          : HttpResponse.json(makePage(firstPage, 1, 10, 11))
      )
    );

    renderComponent('?page_size=10');

    await screen.findByText('job-1', undefined, { timeout: XL_SELECTOR_TIMEOUT });
    await user.click(screen.getByRole('button', { name: /next page/i }));

    expect(await findErrorPanel()).toBeInTheDocument();
    expect(screen.queryByTestId('entity-empty-state-first-use')).not.toBeInTheDocument();
  });

  it('stops polling while the list is in error', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let requestCount = 0;
    server.use(
      http.get(LIST_URL, () => {
        requestCount += 1;
        return failure();
      })
    );

    renderComponent();

    await findErrorPanel();
    const countAtError = requestCount;
    await vi.advanceTimersByTimeAsync(JOB_POLLING_INTERVAL_MS * 3);

    expect(requestCount).toBe(countAtError);
    expect(screen.getByTestId('error-panel')).toBeInTheDocument();
  });

  it('shows the first-use state when the request succeeds with no jobs', async () => {
    server.use(http.get(LIST_URL, () => HttpResponse.json(makePage([]))));

    renderComponent();

    expect(
      await screen.findByText(ENTITY_EMPTY_STATES.anonymizerJobs.heading, undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    expect(screen.queryByTestId('error-panel')).not.toBeInTheDocument();
  });

  it('shows no-results when a filtered search succeeds with no matches', async () => {
    server.use(http.get(LIST_URL, () => HttpResponse.json(makePage([]))));

    renderComponent('?s=nomatch');

    expect(
      await screen.findByTestId('entity-empty-state-no-results', undefined, {
        timeout: XL_SELECTOR_TIMEOUT,
      })
    ).toBeInTheDocument();
    expect(screen.queryByTestId('error-panel')).not.toBeInTheDocument();
  });

  describe('cancel confirmation', () => {
    const CANCEL_URL = `${PLATFORM_BASE_URL}/apis/anonymizer/v2/workspaces/:workspace/jobs/run/:name/cancel`;
    const JOB = 'cancel-me';

    const setup = async (status: HelixJobStatus = HelixJobStatus.pending) => {
      let posts = 0;
      const handler = (respond: (n: number) => Response | Promise<Response>) =>
        http.post(CANCEL_URL, () => {
          posts += 1;
          return respond(posts);
        });
      server.use(
        http.get(LIST_URL, () => HttpResponse.json(makePage([{ ...makeJob(JOB), status }])))
      );
      const user = userEvent.setup();
      renderComponent();
      await screen.findByText(JOB, undefined, { timeout: XL_SELECTOR_TIMEOUT });
      await user.click(screen.getByRole('button', { name: 'Open quick actions menu' }));
      return { user, handler, getPosts: () => posts };
    };

    const selectCancel = async (user: UserEvent) => {
      await user.click(await screen.findByRole('menuitem', { name: 'Cancel' }));
      return screen.findByRole('dialog');
    };

    it('opens the confirmation dialog without sending a POST', async () => {
      const { user, handler, getPosts } = await setup();
      server.use(handler(() => HttpResponse.json({})));

      const dialog = await selectCancel(user);

      expect(within(dialog).getByText(`Cancel ${JOB}`)).toBeInTheDocument();
      expect(getPosts()).toBe(0);
    });

    it('dismissing sends no POST', async () => {
      const { user, handler, getPosts } = await setup();
      server.use(handler(() => HttpResponse.json({})));
      const dialog = await selectCancel(user);

      await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(getPosts()).toBe(0);
    });

    it('sends exactly one POST when confirm is double-clicked, then closes', async () => {
      const { user, handler, getPosts } = await setup();
      server.use(
        handler(async () => {
          await delay(50);
          return HttpResponse.json({ name: JOB, status: HelixJobStatus.cancelled });
        })
      );
      const dialog = await selectCancel(user);

      await user.dblClick(within(dialog).getByRole('button', { name: 'Cancel Job' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(getPosts()).toBe(1);
    });

    it('shows a 409 detail in the open dialog and closes after a successful retry', async () => {
      const { user, handler, getPosts } = await setup();
      server.use(
        handler((n) =>
          n === 1
            ? HttpResponse.json({ detail: 'Job is already finishing' }, { status: 409 })
            : HttpResponse.json({ name: JOB, status: HelixJobStatus.cancelled })
        )
      );
      const dialog = await selectCancel(user);

      await user.click(within(dialog).getByRole('button', { name: 'Cancel Job' }));

      expect(await within(dialog).findByText('Job is already finishing')).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Cancel Job' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(getPosts()).toBe(2);
    });

    it('shows no Cancel item for a completed job (guard)', async () => {
      await setup(HelixJobStatus.completed);

      expect(await screen.findByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
      expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });
  });
});
