// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { getDataDesignerListCreateJobsQueryKey } from '@nemo/sdk/generated/data-designer/data-designer';
import type { CreateJob, CreateJobsPage } from '@nemo/sdk/generated/data-designer/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/data-designer/schema/HelixJobStatus';
import { DataDesignerJobsDataView } from '@studio/components/dataViews/DataDesignerJobsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getDataDesignerJobListRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent, { type UserEvent } from '@testing-library/user-event';
import { delay, http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const LIST_URL = mockApiUrl(getDataDesignerListCreateJobsQueryKey, ':workspace');
const WORKSPACE = workspace1.workspace;
const FAILURE_DETAIL = 'Audit simulated list failure';

const makeJob = (name: string): CreateJob => ({
  id: `${name}-id`,
  name,
  workspace: WORKSPACE,
  status: HelixJobStatus.active,
  spec: {} as CreateJob['spec'],
  created_at: '2025-06-01T10:00:00Z',
});

const makePage = (jobs: CreateJob[], page = 1, pageSize = 25, totalResults = jobs.length) =>
  ({
    data: jobs,
    pagination: {
      page,
      page_size: pageSize,
      current_page_size: jobs.length,
      total_pages: Math.ceil(totalResults / pageSize),
      total_results: totalResults,
    },
  }) satisfies CreateJobsPage;

const failure = () => HttpResponse.json({ detail: FAILURE_DETAIL }, { status: 500 });

const renderComponent = (query = '') =>
  renderRoute(<DataDesignerJobsDataView />, {
    history: `${getDataDesignerJobListRoute(WORKSPACE)}${query}`,
    routes: [{ path: ROUTES.workspace.dataDesignerJobList, element: <DataDesignerJobsDataView /> }],
  });

const findErrorPanel = () =>
  screen.findByTestId('error-panel', undefined, { timeout: XL_SELECTOR_TIMEOUT });

describe('DataDesignerJobsDataView', () => {
  it('shows the list-load error instead of the first-use state when the request fails', async () => {
    server.use(http.get(LIST_URL, failure));

    renderComponent();

    expect(await findErrorPanel()).toBeInTheDocument();
    expect(screen.getByText(FAILURE_DETAIL)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.queryByTestId('entity-empty-state-first-use')).not.toBeInTheDocument();
    expect(
      screen.queryByText(ENTITY_EMPTY_STATES.dataDesignerJobs.heading)
    ).not.toBeInTheDocument();
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

  it('shows the first-use state when the request succeeds with no jobs', async () => {
    server.use(http.get(LIST_URL, () => HttpResponse.json(makePage([]))));

    renderComponent();

    expect(
      await screen.findByText(ENTITY_EMPTY_STATES.dataDesignerJobs.heading, undefined, {
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
    const JOB = 'cancel-me';
    const CANCEL_URL = `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create/:name/cancel`;

    const setup = async (status: HelixJobStatus, onPost: () => Response | Promise<Response>) => {
      const counter = { posts: 0 };
      server.use(
        http.get(LIST_URL, () => HttpResponse.json(makePage([{ ...makeJob(JOB), status }]))),
        http.post(CANCEL_URL, async () => {
          counter.posts += 1;
          return onPost();
        })
      );
      const user = userEvent.setup();
      renderComponent();
      await screen.findByText(JOB, undefined, { timeout: XL_SELECTOR_TIMEOUT });
      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      return { user, counter };
    };
    const ok = () => HttpResponse.json({ name: JOB, status: HelixJobStatus.cancelled });
    const openDialog = async (user: UserEvent) => {
      await user.click(await screen.findByRole('menuitem', { name: 'Cancel' }));
      return screen.findByRole('dialog');
    };

    it('opens the dialog on Cancel and sends no POST', async () => {
      const { user, counter } = await setup(HelixJobStatus.pending, ok);

      const dialog = await openDialog(user);

      expect(within(dialog).getByText(`Cancel ${JOB}`)).toBeInTheDocument();
      expect(counter.posts).toBe(0);
    });

    it('dismissing with Cancel sends no POST', async () => {
      const { user, counter } = await setup(HelixJobStatus.pending, ok);
      const dialog = await openDialog(user);

      await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(counter.posts).toBe(0);
    });

    it('sends exactly one POST on a double click of Cancel Job and closes', async () => {
      const { user, counter } = await setup(HelixJobStatus.pending, async () => {
        await delay(50);
        return ok();
      });
      const dialog = await openDialog(user);

      await user.dblClick(within(dialog).getByRole('button', { name: 'Cancel Job' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(counter.posts).toBe(1);
    });

    it('shows a 409 detail in the open dialog and closes after a successful retry', async () => {
      let n = 0;
      const { user, counter } = await setup(HelixJobStatus.pending, () =>
        ++n === 1
          ? HttpResponse.json({ detail: 'Job is already finishing' }, { status: 409 })
          : ok()
      );
      const dialog = await openDialog(user);

      await user.click(within(dialog).getByRole('button', { name: 'Cancel Job' }));
      expect(await within(dialog).findByText('Job is already finishing')).toBeInTheDocument();
      await user.click(within(dialog).getByRole('button', { name: 'Cancel Job' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(counter.posts).toBe(2);
    });

    it('shows no Cancel item for a completed job', async () => {
      await setup(HelixJobStatus.completed, ok);

      await screen.findByRole('menuitem', { name: /clone/i });
      expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });
  });

  describe('delete confirmation', () => {
    const DELETE_URL = `${PLATFORM_BASE_URL}/apis/data-designer/v2/workspaces/:workspace/jobs/create/:name`;
    const DESCRIPTION_ONE = "The job is removed from this workspace and can't be restored.";
    const DESCRIPTION_MANY = "The jobs are removed from this workspace and can't be restored.";

    const setup = async (names: string[]) => {
      const deleted: string[] = [];
      server.use(
        http.get(LIST_URL, () =>
          HttpResponse.json(
            makePage(names.map((name) => ({ ...makeJob(name), status: HelixJobStatus.completed })))
          )
        ),
        http.delete(DELETE_URL, ({ params }) => {
          deleted.push(String(params.name));
          return new HttpResponse(null, { status: 204 });
        })
      );
      const user = userEvent.setup();
      renderComponent();
      await screen.findByText(names[0], undefined, { timeout: XL_SELECTOR_TIMEOUT });
      return { user, deleted };
    };

    const selectRow = async (user: UserEvent, index: number) => {
      const rowCheckboxAt = () =>
        screen.getAllByRole('checkbox', { name: /(De)?select row/i })[index];
      await waitFor(() => expect(rowCheckboxAt()).toBeEnabled());
      await user.click(rowCheckboxAt());
      await waitFor(() => expect(rowCheckboxAt()).toBeChecked());
    };

    it('names the job from the row Delete action and deletes it on confirm', async () => {
      const { user, deleted } = await setup(['delete-me']);
      await user.click(screen.getAllByRole('button', { name: /actions/i })[0]);
      await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));

      const dialog = await screen.findByRole('dialog');
      expect(
        within(dialog).getByRole('heading', { name: "Delete Data Designer job 'delete-me'?" })
      ).toBeInTheDocument();
      expect(within(dialog).getByText(DESCRIPTION_ONE)).toBeInTheDocument();

      await user.click(within(dialog).getByRole('button', { name: 'Delete Data Designer Job' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect(deleted).toEqual(['delete-me']);
      expect(await screen.findByTestId('mock-toast-success')).toHaveTextContent(
        "Deleted Data Designer job 'delete-me'."
      );
    });

    it('lists every selected job in the bulk dialog and deletes them all', async () => {
      const { user, deleted } = await setup(['job-a', 'job-b']);
      await screen.findByText('job-b');
      await selectRow(user, 0);
      await selectRow(user, 1);
      await user.click(await screen.findByRole('button', { name: 'Delete selected jobs' }));

      const dialog = await screen.findByRole('dialog');
      expect(
        within(dialog).getByRole('heading', { name: 'Delete 2 Data Designer jobs?' })
      ).toBeInTheDocument();
      expect(within(dialog).getByText(DESCRIPTION_MANY)).toBeInTheDocument();
      expect(within(dialog).getByText('job-a')).toBeInTheDocument();
      expect(within(dialog).getByText('job-b')).toBeInTheDocument();

      await user.click(within(dialog).getByRole('button', { name: 'Delete 2 Data Designer Jobs' }));

      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      expect([...deleted].sort()).toEqual(['job-a', 'job-b']);
    });
  });
});
