// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvaluationResultsDataView } from '@studio/components/dataViews/EvaluationResultsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const WORKSPACE = 'default';

const JOBS_URL = `${PLATFORM_BASE_URL}/apis/evaluator/v2/workspaces/:workspace/evaluate/jobs`;

const mockJobsPage = {
  data: [
    {
      id: 'eval-job-1',
      name: 'my-model-eval',
      workspace: WORKSPACE,
      status: 'completed',
      created_at: '2025-01-01T00:00:00Z',
    },
  ],
  pagination: { page: 1, page_size: 50, current_page_size: 1, total_pages: 1, total_results: 1 },
};

const NOT_AGENT_TRIGGERED = {
  $or: [
    { 'spec.target.format': { $nin: ['generic', 'nemo_agent_toolkit'] } },
    { 'spec.target.format': { $eq: null } },
  ],
};

const renderDataView = () =>
  renderRoute(undefined, {
    history: `/workspaces/${WORKSPACE}/evaluation/results`,
    routes: [{ path: ROUTES.workspace.evaluationResults, element: <EvaluationResultsDataView /> }],
  });

describe('EvaluationResultsDataView filter', () => {
  let sentFilters: Array<string | null>;

  beforeEach(() => {
    sentFilters = [];
    server.use(
      http.get(JOBS_URL, ({ request }) => {
        sentFilters.push(new URL(request.url).searchParams.get('filter'));
        return HttpResponse.json(mockJobsPage);
      })
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it('excludes agent-triggered jobs by target format', async () => {
    renderDataView();

    await waitFor(() => expect(sentFilters.length).toBeGreaterThan(0));
    expect(JSON.parse(sentFilters[0] as string)).toEqual(NOT_AGENT_TRIGGERED);
  });

  it('keeps the exclusion when the user searches by name', async () => {
    renderDataView();
    await waitFor(() => expect(sentFilters.length).toBeGreaterThan(0));

    await userEvent.type(await screen.findByPlaceholderText('Search by name'), 'foo');

    await waitFor(() => {
      const latest = JSON.parse(sentFilters[sentFilters.length - 1] as string);
      expect(latest).toEqual({ $and: [NOT_AGENT_TRIGGERED, { name: { $like: 'foo' } }] });
    });
  });
});
