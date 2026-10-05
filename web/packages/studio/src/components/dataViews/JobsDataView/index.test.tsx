// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import {
  HelixJobResponse,
  HelixJobResponsesPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { JobsDataView } from '@studio/components/dataViews/JobsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { server } from '@studio/mocks/node';
import { getWorkspaceJobsRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.hoisted(() => {
  vi.stubEnv('VITE_FF_CUSTOMIZER_ENABLED', 'false');
});

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/jobs/v2/workspaces/:workspace/jobs`;
const WORKSPACE = workspace1.workspace;

const makeJob = (overrides: Partial<HelixJobResponse> = {}): HelixJobResponse => ({
  id: 'job-id-1',
  attempt_id: 'attempt-1',
  name: 'my-training-job',
  workspace: WORKSPACE,
  source: 'evaluator-metrics',
  fileset: 'fileset-1',
  status: HelixJobStatus.completed,
  platform_spec: { steps: [] },
  created_at: '2025-06-01T10:00:00Z',
  updated_at: '2025-06-01T12:00:00Z',
  ...overrides,
});

const makeJobsPage = (jobs: HelixJobResponse[]): HelixJobResponsesPage => ({
  data: jobs,
  pagination: {
    page: 1,
    page_size: 25,
    current_page_size: jobs.length,
    total_pages: 1,
    total_results: jobs.length,
  },
});

const renderComponent = () =>
  renderRoute(<JobsDataView />, {
    history: getWorkspaceJobsRoute(WORKSPACE),
    routes: [
      {
        path: ROUTES.workspace.jobs,
        element: <JobsDataView />,
      },
    ],
  });

describe('JobsDataView', () => {
  it('shows empty state when there are no jobs', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([]))));

    renderComponent();

    expect(await screen.findByText(ENTITY_EMPTY_STATES.jobs.heading)).toBeInTheDocument();
  });

  it('renders job data in the table', async () => {
    const jobs = [
      makeJob({ name: 'data-designer-run-1', source: 'data-designer' }),
      makeJob({ name: 'eval-run-2', source: 'evaluator-metrics', id: 'job-id-2' }),
    ];
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage(jobs))));

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('data-designer-run-1')).toBeInTheDocument();
      expect(screen.getByText('eval-run-2')).toBeInTheDocument();
    });
  });

  it('names the source the same way in the tag and its tooltip', async () => {
    // The tooltip otherwise falls back to the raw accessor value — the API's own word for
    // the service — so a row reads one thing and hovers another. Checked on data-designer
    // because this file disables the customizer flag, which filters those rows out.
    const jobs = [makeJob({ name: 'data-designer-run-1', source: 'data-designer' })];
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage(jobs))));

    renderComponent();

    await screen.findByText('data-designer-run-1');

    expect(screen.getByTitle('Data Designer')).toBeInTheDocument();
    expect(screen.queryByTitle('data-designer')).not.toBeInTheDocument();
  });

  it('hides customizer jobs when customizer is disabled', async () => {
    const jobs = [
      makeJob({ name: 'customizer-run-1', source: 'customization' }),
      makeJob({ name: 'eval-run-2', source: 'evaluator-metrics', id: 'job-id-2' }),
    ];
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage(jobs))));

    renderComponent();

    expect(await screen.findByText('eval-run-2')).toBeInTheDocument();
    expect(screen.queryByText('customizer-run-1')).not.toBeInTheDocument();
    expect(screen.queryByText('Fine-tuning')).not.toBeInTheDocument();
  });

  it('renders expected column headers', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([makeJob()]))));

    renderComponent();

    for (const header of ['Name', 'Source', 'Status', 'Created']) {
      expect(await screen.findByRole('columnheader', { name: header })).toBeInTheDocument();
    }
  });

  it('adds a returned plugin source to the source filter options', async () => {
    const jobs = [
      makeJob({ name: 'agents-run-1', source: 'nemo-agents-plugin' }),
      makeJob({ name: 'eval-run-2', source: 'evaluator-metrics', id: 'job-id-2' }),
    ];
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage(jobs))));

    renderComponent();

    fireEvent.click(await screen.findByTestId('open-filters-button'));
    fireEvent.click(await screen.findByTestId('column-filter-source'));

    expect(await screen.findByRole('option', { name: 'nemo-agents-plugin' })).toBeInTheDocument();
  });

  describe('source tag', () => {
    it('renders the source value as a selectable, focusable tag button', async () => {
      server.use(
        http.get(JOBS_URL, () =>
          HttpResponse.json(makeJobsPage([makeJob({ source: 'evaluator-metrics' })]))
        )
      );

      renderComponent();

      const tag = await screen.findByRole('button', { name: 'Evaluator' });
      expect(tag.tagName).toBe('BUTTON');

      tag.focus();
      expect(tag).toHaveFocus();
    });

    it('applies the source filter when the source tag is clicked', async () => {
      const requestUrls: string[] = [];
      server.use(
        http.get(JOBS_URL, ({ request }) => {
          requestUrls.push(request.url);
          return HttpResponse.json(makeJobsPage([makeJob({ source: 'evaluator-metrics' })]));
        })
      );
      const user = userEvent.setup();

      renderComponent();

      await user.click(await screen.findByRole('button', { name: 'Evaluator' }));

      await waitFor(() => {
        const matched = requestUrls.find((u) => new URL(u).searchParams.has('filter[source]'));
        expect(matched).toBeDefined();
      });
      const matchedUrl = requestUrls.find((u) => new URL(u).searchParams.has('filter[source]'))!;
      expect(new URL(matchedUrl).searchParams.get('filter[source]')).toBe('evaluator-metrics');
    });

    it('applies the source filter when the source tag is activated via keyboard', async () => {
      const requestUrls: string[] = [];
      server.use(
        http.get(JOBS_URL, ({ request }) => {
          requestUrls.push(request.url);
          return HttpResponse.json(makeJobsPage([makeJob({ source: 'evaluator-metrics' })]));
        })
      );
      const user = userEvent.setup();

      renderComponent();

      const tag = await screen.findByRole('button', { name: 'Evaluator' });
      tag.focus();
      expect(tag).toHaveFocus();
      await user.keyboard('{Enter}');

      await waitFor(() => {
        const matched = requestUrls.find((u) => new URL(u).searchParams.has('filter[source]'));
        expect(matched).toBeDefined();
      });
      const matchedUrl = requestUrls.find((u) => new URL(u).searchParams.has('filter[source]'))!;
      expect(new URL(matchedUrl).searchParams.get('filter[source]')).toBe('evaluator-metrics');
    });
  });

  it('shows error panel when API returns an error', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.error()));

    renderComponent();

    expect(await screen.findByTestId('error-panel')).toBeInTheDocument();
  });

  it('shows search input with correct placeholder', async () => {
    server.use(http.get(JOBS_URL, () => HttpResponse.json(makeJobsPage([]))));

    renderComponent();

    expect(await screen.findByPlaceholderText('Search by name')).toBeInTheDocument();
  });

  describe('filter query params', () => {
    const captureRequests = () => {
      const requestUrls: string[] = [];
      server.use(
        http.get(JOBS_URL, ({ request }) => {
          requestUrls.push(request.url);
          return HttpResponse.json(makeJobsPage([]));
        })
      );
      return requestUrls;
    };

    const renderWithQuery = (query: string) =>
      renderRoute(<JobsDataView />, {
        history: `${getWorkspaceJobsRoute(WORKSPACE)}${query}`,
        routes: [{ path: ROUTES.workspace.jobs, element: <JobsDataView /> }],
      });

    it('hides system and customizer jobs by default via filter[source][$nin]', async () => {
      const requestUrls = captureRequests();
      renderWithQuery('');

      await waitFor(() => expect(requestUrls.length).toBeGreaterThan(0));

      const params = new URL(requestUrls.at(-1)!).searchParams;
      expect(params.getAll('filter[source][$nin]').join(',')).toContain('models-system');
      expect(params.getAll('filter[source][$nin]').join(',')).toContain('customization');
      const keys = Array.from(params.keys());
      expect(keys.some((k) => k === 'search' || k.startsWith('search['))).toBe(false);
    });

    it('omits the $nin source filter when user picks an explicit source', async () => {
      const requestUrls = captureRequests();
      const filters = encodeURIComponent(
        JSON.stringify([{ id: 'source', value: 'evaluator-metrics' }])
      );
      renderWithQuery(`?filters=${filters}`);

      let matchedUrl: string | undefined;
      await waitFor(() => {
        matchedUrl = requestUrls.find((u) => new URL(u).searchParams.has('filter[source]'));
        expect(matchedUrl).toBeDefined();
      });

      const params = new URL(matchedUrl!).searchParams;
      expect(params.get('filter[source]')).toBe('evaluator-metrics');
      expect(params.has('filter[source][$nin]')).toBe(false);
    });

    it('drops a stale customizer source filter when customizer is disabled', async () => {
      const requestUrls = captureRequests();
      const filters = encodeURIComponent(
        JSON.stringify([{ id: 'source', value: 'customization' }])
      );
      renderWithQuery(`?filters=${filters}`);

      await waitFor(() => expect(requestUrls.length).toBeGreaterThan(0));

      const params = new URL(requestUrls.at(-1)!).searchParams;
      expect(params.has('filter[source]')).toBe(false);
      expect(params.getAll('filter[source][$nin]').join(',')).toContain('customization');
    });

    it('sends filter[name][$like] when name search is active', async () => {
      const requestUrls = captureRequests();
      renderWithQuery('?s=foo');

      let matchedUrl: string | undefined;
      await waitFor(() => {
        matchedUrl = requestUrls.find((u) => new URL(u).searchParams.has('filter[name][$like]'));
        expect(matchedUrl).toBeDefined();
      });

      expect(new URL(matchedUrl!).searchParams.get('filter[name][$like]')).toBe('foo');
    });

    it('never emits a top-level search or search[ key', async () => {
      const requestUrls = captureRequests();
      renderWithQuery('?s=foo');

      await waitFor(() => expect(requestUrls.length).toBeGreaterThan(0));

      for (const raw of requestUrls) {
        const params = new URL(raw).searchParams;
        expect(params.has('search')).toBe(false);
        expect(Array.from(params.keys()).some((k) => k.startsWith('search['))).toBe(false);
      }
    });

    it('sends sort=source when the Source header is clicked', async () => {
      const requestUrls: string[] = [];
      server.use(
        http.get(JOBS_URL, ({ request }) => {
          requestUrls.push(request.url);
          return HttpResponse.json(makeJobsPage([makeJob()]));
        })
      );
      renderComponent();

      fireEvent.click(await screen.findByRole('button', { name: /^Source/ }));

      await waitFor(() =>
        expect(requestUrls.some((u) => new URL(u).searchParams.get('sort') === 'source')).toBe(true)
      );
    });
  });
});
