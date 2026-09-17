// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import { mockTracesPage } from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { EvaluationSessionDetailRoute } from '@studio/routes/EvaluationSessionDetailRoute';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { http, HttpResponse } from 'msw';

const renderSessionDetail = (search = '') =>
  renderRoute(undefined, {
    history: `/workspaces/default/experiment/my-group/my-experiment/sessions/session-agent-run-001${search}`,
    routes: [
      {
        path: '/workspaces/:workspace/experiment/:experimentName/:evaluationName/sessions/:sessionId',
        element: <EvaluationSessionDetailRoute />,
      },
    ],
  });

describe('EvaluationSessionDetailRoute', () => {
  it('renders an evaluation session summary', async () => {
    const traceModes: Array<string | null> = [];
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
        const url = new URL(request.url);
        traceModes.push(url.searchParams.get('mode'));
        const sessionId = url.searchParams.get('filter[session_id]');
        const data = mockTracesPage.data
          .filter((trace) => trace.session_id === sessionId)
          .map((trace) => ({ ...trace, experiment_context: undefined }));
        return HttpResponse.json({
          ...mockTracesPage,
          data,
          pagination: {
            ...mockTracesPage.pagination,
            current_page_size: data.length,
            total_results: data.length,
          },
        });
      })
    );

    renderSessionDetail();

    expect(await screen.findByText('Test case: case-0042')).toBeInTheDocument();
    expect(screen.queryByText('2 traces')).not.toBeInTheDocument();
    await waitFor(() => expect(traceModes).toEqual(['summary']));
  });

  it('retains evaluation context while rendering a selected trace', async () => {
    renderSessionDetail('?traceId=trace-agent-run-001');

    expect(await screen.findByText('Test case: case-0042')).toBeInTheDocument();
    expect(screen.getByText('Evaluation Context')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Intake' })).not.toBeInTheDocument();
  });

  it('offers the "Compare against evaluation run" entry point on the single session view', async () => {
    renderSessionDetail();

    // The selector enables once the test case's sibling runs resolve.
    const trigger = await screen.findByRole('combobox', {
      name: 'Compare against evaluation run',
    });
    await waitFor(() => expect(trigger).not.toHaveAttribute('data-disabled'));
  });
});
