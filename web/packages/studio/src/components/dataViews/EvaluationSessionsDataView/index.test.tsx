// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvaluationSessionsDataView } from '@studio/components/dataViews/EvaluationSessionsDataView';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { server } from '@studio/mocks/node';
import { getEvaluationDetailRoute } from '@studio/routes/utils';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('use-debounce', () => ({
  useDebounce: (value: unknown) => [value, { cancel: () => {}, flush: () => {} }],
}));

const WORKSPACE = 'default';
const EXPERIMENT = 'my-group';
const EXPERIMENT_NAME = 'my-experiment';
const TRACE_ID = 'trace-abc-123';

const SESSIONS_URL = `${PLATFORM_BASE_URL}/apis/intake/v2/workspaces/:workspace/evaluations/:name/sessions`;
const EVALUATION_URL = `${PLATFORM_BASE_URL}/apis/intake/v2/workspaces/:workspace/evaluations/:name`;

const mockSession = {
  workspace: WORKSPACE,
  experiment_name: EXPERIMENT_NAME,
  session_id: 'session-1',
  trace_id: TRACE_ID,
  root_span_id: 'span-root-1',
  started_at: '2025-01-01T00:00:00Z',
  status: 'success',
  test_case_name: 'case-1',
  input: 'Session input',
  output: 'Session output',
};

const mockSessionsPage = {
  data: [mockSession],
  pagination: { page: 1, page_size: 50, current_page_size: 1, total_pages: 1, total_results: 1 },
};

const mockEvaluation = {
  workspace: WORKSPACE,
  name: EXPERIMENT_NAME,
  experiment_group_name: EXPERIMENT,
  run_count: 1,
  evaluator_names: [],
};

const renderDataView = () =>
  renderRoute(undefined, {
    history: getEvaluationDetailRoute(WORKSPACE, EXPERIMENT, EXPERIMENT_NAME),
    routes: [
      {
        path: ROUTES.workspace.evaluationDetail,
        element: (
          <EvaluationSessionsDataView
            evaluationName={EXPERIMENT_NAME}
            experimentName={EXPERIMENT}
          />
        ),
      },
      {
        path: ROUTES.workspace.evaluationSessionDetail,
        element: <div data-testid="trace-detail-route" />,
      },
    ],
  });

describe('EvaluationSessionsDataView', () => {
  let sessionRequestModes: Array<string | null>;

  beforeEach(() => {
    sessionRequestModes = [];
    server.use(
      http.get(EVALUATION_URL, () => HttpResponse.json(mockEvaluation)),
      http.get(SESSIONS_URL, ({ request }) => {
        sessionRequestModes.push(new URL(request.url).searchParams.get('mode'));
        return HttpResponse.json(mockSessionsPage);
      })
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it('renders a row for each session', async () => {
    renderDataView();
    // The Tooltip renders both a trigger and a hidden popover — getAllByText handles both.
    const matches = await screen.findAllByText('case-1');
    expect(matches.length).toBeGreaterThan(0);
    expect(screen.getByText('Session input')).toBeInTheDocument();
    expect(screen.getByText('Session output')).toBeInTheDocument();
  });

  it('marks an evaluator that ran but produced no score as failed', async () => {
    server.use(
      http.get(EVALUATION_URL, () =>
        HttpResponse.json({ ...mockEvaluation, evaluator_names: ['judge.accuracy'] })
      ),
      http.get(SESSIONS_URL, () =>
        HttpResponse.json({
          ...mockSessionsPage,
          data: [{ ...mockSession, evaluator_scores: {}, failed_evaluators: ['judge.accuracy'] }],
        })
      )
    );
    renderDataView();
    expect((await screen.findAllByText('Failed')).length).toBeGreaterThan(0);
  });

  it('requests experiment sessions in preview mode', async () => {
    renderDataView();

    await screen.findAllByText('case-1');

    await waitFor(() => expect(sessionRequestModes).toContain('preview'));
    expect(sessionRequestModes).not.toContain('detailed');
  });

  it('falls back without mode when the backend rejects the preview query parameter', async () => {
    server.use(
      http.get(SESSIONS_URL, ({ request }) => {
        const mode = new URL(request.url).searchParams.get('mode');
        sessionRequestModes.push(mode);
        if (mode === 'preview') {
          return HttpResponse.json(
            { detail: 'Unsupported query parameter(s): mode' },
            { status: 400 }
          );
        }
        return HttpResponse.json(mockSessionsPage);
      })
    );

    renderDataView();

    await screen.findAllByText('case-1');

    await waitFor(() => expect(sessionRequestModes).toEqual(['preview', null]));
  });

  it('navigates to the experiment session route with the trace selected', async () => {
    const user = userEvent.setup();
    renderDataView();

    // Click the tooltip trigger (first match); the row click handler delegates up from there.
    const [trigger] = await screen.findAllByText('case-1');
    await user.click(trigger);

    expect(await screen.findByTestId('trace-detail-route')).toBeInTheDocument();
  });

  it('does not navigate when the session has no trace_id', async () => {
    server.use(
      http.get(SESSIONS_URL, () =>
        HttpResponse.json({
          ...mockSessionsPage,
          data: [{ ...mockSession, trace_id: '', test_case_name: 'no-trace-case' }],
        })
      )
    );
    const user = userEvent.setup();
    renderDataView();

    const [trigger] = await screen.findAllByText('no-trace-case');
    await user.click(trigger);

    // Route should not have changed — trace-detail-route element is absent.
    expect(screen.queryByTestId('trace-detail-route')).not.toBeInTheDocument();
  });
});
