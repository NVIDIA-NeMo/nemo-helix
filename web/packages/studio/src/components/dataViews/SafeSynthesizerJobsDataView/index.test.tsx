// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { getSafeSynthesizerListJobsQueryKey } from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import {
  type GenerateJob,
  type GenerateJobsPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/safe-synthesizer/schema';
import { GenerateJobsDataView } from '@studio/components/dataViews/SafeSynthesizerJobsDataView';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getWorkspaceSafeSynthesizerRoute } from '@studio/routes/utils';
import { XL_SELECTOR_TIMEOUT } from '@studio/tests/util/constants';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const LIST_URL = mockApiUrl(getSafeSynthesizerListJobsQueryKey, ':workspace');
const WORKSPACE = workspace1.workspace;
const FAILURE_DETAIL = 'Audit simulated list failure';

const makeJob = (name: string): GenerateJob => ({
  id: `${name}-id`,
  name,
  workspace: WORKSPACE,
  status: HelixJobStatus.active,
  spec: { data_source: `fileset://${WORKSPACE}/dataset/source.csv`, config: {} },
  created_at: '2025-06-01T10:00:00Z',
});

const makePage = (jobs: GenerateJob[], page = 1, pageSize = 25, totalResults = jobs.length) =>
  ({
    data: jobs,
    pagination: {
      page,
      page_size: pageSize,
      current_page_size: jobs.length,
      total_pages: Math.ceil(totalResults / pageSize),
      total_results: totalResults,
    },
  }) satisfies GenerateJobsPage;

const failure = () => HttpResponse.json({ detail: FAILURE_DETAIL }, { status: 500 });

const renderComponent = (query = '') =>
  renderRoute(<GenerateJobsDataView />, {
    history: `${getWorkspaceSafeSynthesizerRoute(WORKSPACE)}${query}`,
    routes: [{ path: ROUTES.workspace.safeSynthesizer, element: <GenerateJobsDataView /> }],
  });

const findErrorPanel = () =>
  screen.findByTestId('error-panel', undefined, { timeout: XL_SELECTOR_TIMEOUT });

describe('GenerateJobsDataView', () => {
  it('shows the list-load error instead of the first-use state when the request fails', async () => {
    server.use(http.get(LIST_URL, failure));

    renderComponent();

    expect(await findErrorPanel()).toBeInTheDocument();
    expect(screen.getByText(FAILURE_DETAIL)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    expect(screen.queryByTestId('entity-empty-state-first-use')).not.toBeInTheDocument();
    expect(
      screen.queryByText(ENTITY_EMPTY_STATES.safeSynthesizerJobs.heading)
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
      await screen.findByText(ENTITY_EMPTY_STATES.safeSynthesizerJobs.heading, undefined, {
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
});
