// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getGetSessionQueryKey } from '@nemo/sdk/generated/platform/sessions';
import { getGetSpanQueryKey, getListSpansQueryKey } from '@nemo/sdk/generated/platform/spans';
import { getGetTraceQueryKey, getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import {
  mockSessionById,
  mockSpanById,
  mockSpansPage,
  mockTracesPage,
} from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { IntakeSessionDetailRoute } from '@studio/routes/IntakeSessionDetailRoute';
import { mockFeatureFlags } from '@studio/tests/util/mockFeatureFlags';
import { renderRoute, screen, waitFor, within } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';

const LocationProbe = () => {
  const location = useLocation();
  return <output data-testid="location">{`${location.pathname}${location.search}`}</output>;
};

const renderSessionDetail = (sessionId = 'session-agent-run-001', search = '') =>
  renderRoute(undefined, {
    history: `/workspaces/default/intake/sessions/${sessionId}${search}`,
    routes: [
      {
        path: '/workspaces/:workspace/intake/sessions/:sessionId',
        element: (
          <>
            <LocationProbe />
            <IntakeSessionDetailRoute />
          </>
        ),
      },
    ],
  });

describe('IntakeSessionDetailRoute', () => {
  afterEach(() => sessionStorage.clear());

  it('hydrates the session header and all trace trajectories with summary payloads', async () => {
    const sessionDetailRequests: string[] = [];
    const traceListRequests: URL[] = [];
    const traceDetailRequests: string[] = [];
    const spanListRequests: URL[] = [];
    const spanDetailRequests: string[] = [];
    server.use(
      http.get(mockApiUrl(getGetSessionQueryKey, ':workspace', ':sessionId'), ({ params }) => {
        const sessionId = String(params['sessionId']);
        sessionDetailRequests.push(sessionId);
        const session = mockSessionById(sessionId);
        return session ? HttpResponse.json(session) : new HttpResponse(null, { status: 404 });
      }),
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
        const url = new URL(request.url);
        traceListRequests.push(url);
        const data = mockTracesPage.data
          .filter((trace) => trace.session_id === url.searchParams.get('filter[session_id]'))
          .sort((a, b) => Date.parse(a.started_at) - Date.parse(b.started_at));
        return HttpResponse.json({
          ...mockTracesPage,
          data,
          pagination: {
            ...mockTracesPage.pagination,
            current_page_size: data.length,
            total_results: data.length,
          },
        });
      }),
      http.get(mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'), ({ params }) => {
        traceDetailRequests.push(String(params['traceId']));
        return new HttpResponse(null, { status: 500 });
      }),
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        const url = new URL(request.url);
        spanListRequests.push(url);
        const data = mockSpansPage.data.filter(
          (span) => span.session_id === url.searchParams.get('filter[session_id]')
        );
        return HttpResponse.json({
          ...mockSpansPage,
          data,
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: data.length,
            total_results: data.length,
          },
        });
      }),
      http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), ({ params }) => {
        spanDetailRequests.push(String(params['spanId']));
        return new HttpResponse(null, { status: 500 });
      })
    );

    renderSessionDetail();

    expect(await screen.findByText('Session session-agent-run-001')).toBeInTheDocument();
    expect(screen.queryByText('2 traces')).not.toBeInTheDocument();
    const sessionSummary = screen.getByTestId('session-summary-header');
    expect(within(sessionSummary).getByText('6')).toBeInTheDocument();
    expect(within(sessionSummary).getByText('2,948')).toBeInTheDocument();
    expect(screen.getByText('Tree')).toBeInTheDocument();
    expect(screen.getByText('List')).toBeInTheDocument();
    expect(screen.queryByText('Graph')).not.toBeInTheDocument();
    expect(
      screen.queryByText('Can I deploy this model in a private workspace?')
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText('How do I restrict access to that private workspace?')
    ).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /View trace/ })).not.toBeInTheDocument();
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    expect(within(trajectory).getByText('Session')).toBeInTheDocument();
    expect(
      within(trajectory).getAllByText('Answer customer policy question').length
    ).toBeGreaterThan(0);
    expect(
      within(trajectory).getAllByText('Explain private workspace access controls').length
    ).toBeGreaterThan(0);
    expect(within(trajectory).getByText('Generate final response')).toBeInTheDocument();
    expect(within(trajectory).getByText('Generate access-control guidance')).toBeInTheDocument();

    await waitFor(() => expect(traceListRequests).toHaveLength(1));
    expect(sessionDetailRequests).toEqual(['session-agent-run-001']);
    expect(traceListRequests[0].searchParams.get('filter[session_id]')).toBe(
      'session-agent-run-001'
    );
    expect(traceListRequests[0].searchParams.get('mode')).toBe('summary');
    expect(traceListRequests[0].searchParams.get('page_size')).toBe('1000');
    expect(traceListRequests[0].searchParams.get('sort')).toBe('started_at');
    expect(traceDetailRequests).toEqual([]);
    await waitFor(() => expect(spanListRequests).toHaveLength(1));
    expect(spanListRequests[0].searchParams.get('filter[trace_id]')).toBeNull();
    expect(spanListRequests[0].searchParams.get('filter[session_id]')).toBe(
      'session-agent-run-001'
    );
    expect(spanListRequests[0].searchParams.get('mode')).toBe('summary');
    expect(spanListRequests[0].searchParams.get('page_size')).toBe('1000');
    expect(spanDetailRequests).toEqual([]);
  });

  it('shows trace metrics when a trace is selected', async () => {
    const sessionDetailRequests: string[] = [];
    const session = mockSessionById('session-agent-run-001');
    expect(session).toBeDefined();
    server.use(
      http.get(mockApiUrl(getGetSessionQueryKey, ':workspace', ':sessionId'), ({ params }) => {
        sessionDetailRequests.push(String(params['sessionId']));
        return HttpResponse.json({ ...session!, span_count: 47, total_tokens: 9999 });
      })
    );

    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    const traceSummary = await screen.findByTestId('session-summary-header');
    expect(within(traceSummary).getByText('4')).toBeInTheDocument();
    expect(within(traceSummary).getByText('1,754')).toBeInTheDocument();
    expect(within(traceSummary).queryByText('47')).not.toBeInTheDocument();
    expect(sessionDetailRequests).toEqual(['session-agent-run-001']);
  });

  it('navigates session to trace to span and back to the session summary', async () => {
    const user = userEvent.setup();
    renderSessionDetail();

    await screen.findByText('Session session-agent-run-001');
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    await user.click(
      within(trajectory).getAllByText('Explain private workspace access controls')[0]
    );

    expect(screen.getByText('Trace Explain private workspace access controls')).toBeInTheDocument();
    expect(screen.getAllByTestId('session-summary-header')).toHaveLength(1);
    expect(within(screen.getByTestId('session-summary-header')).getByText('2')).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent(
      '/workspaces/default/intake/sessions/session-agent-run-001?traceId=trace-agent-run-003'
    );

    await user.click(await screen.findByText('Generate access-control guidance'));
    expect(screen.getByTestId('location')).toHaveTextContent(
      '?traceId=trace-agent-run-003&spanId=span-llm-003'
    );

    await user.click(screen.getByTitle('View session'));
    expect(screen.queryByRole('link', { name: /View trace/ })).not.toBeInTheDocument();
    const restoredTrajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    expect(
      await within(restoredTrajectory).findByText('Generate final response')
    ).toBeInTheDocument();
    expect(screen.getByTestId('location')).toHaveTextContent(
      '/workspaces/default/intake/sessions/session-agent-run-001'
    );
    expect(within(screen.getByTestId('session-summary-header')).getByText('6')).toBeInTheDocument();
  });

  it('deep-links a span from any expanded trace in the session summary', async () => {
    const user = userEvent.setup();
    renderSessionDetail();

    await screen.findByText('Session session-agent-run-001');
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    await user.click(await within(trajectory).findByText('Generate access-control guidance'));

    expect(screen.getByTestId('location')).toHaveTextContent(
      '?traceId=trace-agent-run-003&spanId=span-llm-003'
    );
    expect(
      await within(screen.getByTestId('trace-trajectory-sidebar')).findByText(
        'Generate access-control guidance'
      )
    ).toBeInTheDocument();
    expect(screen.getByText('Trace Explain private workspace access controls')).toBeInTheDocument();
    expect(screen.getByTestId('trace-trajectory-sidebar')).toBeInTheDocument();
  });

  it('keeps the trajectory sidebar mounted while span summaries load and fail', async () => {
    // Hold the spans request open so the loading state is observed deterministically
    // (rather than racing a fixed delay that can unmount the spinner mid-assertion),
    // then fail it explicitly.
    let failSpans = () => {};
    const spansResponse = new Promise<Response>((resolve) => {
      failSpans = () =>
        resolve(HttpResponse.json({ detail: 'Could not load span summaries' }, { status: 500 }));
    });
    server.use(http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), () => spansResponse));

    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    // Sidebar mounts immediately and the spinner shows while spans load.
    expect(await screen.findByLabelText('Loading spans')).toBeInTheDocument();
    expect(screen.getByTestId('trace-trajectory-sidebar')).toBeInTheDocument();

    failSpans();

    // After the failure the error shows and the sidebar is still mounted.
    expect(await screen.findByText('Could not load span summaries')).toBeInTheDocument();
    expect(screen.getByTestId('trace-trajectory-sidebar')).toBeInTheDocument();
  });

  it('does not promote a child span error to the trace-level banner', async () => {
    const erroredChild = {
      ...mockSpanById('span-llm-001')!,
      status: 'error' as const,
      error_type: 'ChildSpanError',
      error_message: 'Only the child span failed.',
    };
    const spanDetailRequests: string[] = [];
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockSpansPage,
          data: [erroredChild],
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: 1,
            total_results: 1,
          },
        })
      ),
      http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), ({ params }) => {
        spanDetailRequests.push(String(params['spanId']));
        return HttpResponse.json(erroredChild);
      })
    );

    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    await waitFor(() => expect(spanDetailRequests).toContain(erroredChild.span_id));
    await waitFor(() => {
      expect(screen.getAllByText('Only the child span failed.')).toHaveLength(1);
    });
  });

  it('keeps the root failure visible while a successful child is selected', async () => {
    const rootSpan = mockSpanById('span-root-002')!;
    const successfulChild = {
      ...mockSpanById('span-llm-001')!,
      span_id: 'span-child-after-root-error',
      session_id: rootSpan.session_id,
      trace_id: rootSpan.trace_id,
      parent_span_id: rootSpan.span_id,
      name: 'Format available troubleshooting context',
    };
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockSpansPage,
          data: [rootSpan, successfulChild],
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: 2,
            total_results: 2,
          },
        })
      ),
      http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), ({ params }) => {
        if (params['spanId'] === successfulChild.span_id) {
          return HttpResponse.json(successfulChild);
        }
        return HttpResponse.json(rootSpan);
      })
    );

    renderSessionDetail(
      rootSpan.session_id,
      `?traceId=${rootSpan.trace_id}&spanId=${successfulChild.span_id}`
    );

    expect(await screen.findByText('Trace failed')).toBeInTheDocument();
    expect(screen.getByText('Knowledge base lookup timed out after 5s.')).toBeInTheDocument();
    expect(screen.getAllByText('Format available troubleshooting context')).not.toHaveLength(0);
  });

  it('preserves the session view mode when selecting a trace', async () => {
    const user = userEvent.setup();
    renderSessionDetail();

    await screen.findByText('Session session-agent-run-001');
    await user.click(screen.getByText('List'));
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    await user.click(
      within(trajectory).getAllByText('Explain private workspace access controls')[0]
    );

    await waitFor(() => {
      expect(screen.getByTestId('location')).toHaveTextContent('?traceId=trace-agent-run-003');
    });
    expect(screen.queryByTestId('trace-trajectory-sidebar')).not.toBeInTheDocument();
    expect(await screen.findByText('Generate access-control guidance')).toBeInTheDocument();
  });

  it('switches a selected trace between graph modes', async () => {
    mockFeatureFlags({ traceGraphEnabled: true });
    const user = userEvent.setup();
    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    await screen.findByText('Trace Answer customer policy question');
    await user.click(screen.getByText('Graph'));

    expect(await screen.findByText('Grouped')).toBeInTheDocument();
    expect(screen.getByText('All spans')).toBeInTheDocument();
    expect(screen.getByText('2 groups from 2 spans')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Longest path' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Most tokens' })).not.toBeInTheDocument();

    await user.click(screen.getByText('All spans'));
    expect(screen.getByText('2 spans and 1 parent link')).toBeInTheDocument();
    const longestPath = screen.getByRole('button', { name: 'Longest path' });
    expect(longestPath).toHaveAttribute('aria-pressed', 'false');
    await user.click(longestPath);
    expect(longestPath).toHaveAttribute('aria-pressed', 'true');

    const mostTokens = screen.getByRole('button', { name: 'Most tokens' });
    expect(mostTokens).toHaveAttribute('aria-pressed', 'false');
    await user.click(mostTokens);
    await waitFor(() => expect(mostTokens).toHaveAttribute('aria-pressed', 'true'));
    expect(screen.getByTestId('location')).toHaveTextContent('spanId=span-llm-001');
  });

  it('loads graph spans by trace when the session page does not contain them', async () => {
    mockFeatureFlags({ traceGraphEnabled: true });
    const traceId = 'trace-agent-run-003';
    const traceSpans = mockSpansPage.data.filter((span) => span.trace_id === traceId);
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        const requestedTraceId = new URL(request.url).searchParams.get('filter[trace_id]');
        const data = requestedTraceId === traceId ? traceSpans : [];
        return HttpResponse.json({
          ...mockSpansPage,
          data,
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: data.length,
            total_results: data.length,
          },
        });
      })
    );
    const user = userEvent.setup();
    renderSessionDetail('session-agent-run-001', `?traceId=${traceId}`);

    const trajectory = await screen.findByRole('navigation', { name: 'Trace trajectory' });
    expect(within(trajectory).getByText('Generate access-control guidance')).toBeInTheDocument();

    await user.click(await screen.findByText('Graph'));

    expect(await screen.findByText('2 groups from 2 spans')).toBeInTheDocument();
  });

  it('warns when the trace span request fails and session spans are used', async () => {
    mockFeatureFlags({ traceGraphEnabled: true });
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        const traceId = new URL(request.url).searchParams.get('filter[trace_id]');
        return traceId ? new HttpResponse(null, { status: 500 }) : HttpResponse.json(mockSpansPage);
      })
    );

    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    expect(
      await screen.findByText(
        'Could not load all spans for this trace. Showing the spans already loaded for the session.'
      )
    ).toBeInTheDocument();
  });

  it('restores graph mode separately for each trace', async () => {
    mockFeatureFlags({ traceGraphEnabled: true });
    const user = userEvent.setup();
    sessionStorage.setItem('nemo-studio:trace-graph-mode:default:trace-agent-run-001', 'all');
    sessionStorage.setItem('nemo-studio:trace-graph-mode:default:trace-agent-run-003', 'grouped');
    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    await user.click(await screen.findByText('Graph'));
    expect(screen.getByRole('radio', { name: 'All spans' })).toBeChecked();

    await user.click(screen.getByText('Tree'));
    await user.click(screen.getByTitle('View session'));
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    await user.click(
      within(trajectory).getAllByText('Explain private workspace access controls')[0]
    );

    await user.click(await screen.findByText('Graph'));
    expect(screen.getByRole('radio', { name: 'Grouped' })).toBeChecked();
  });

  it('hides graph mode when its feature flag is disabled', async () => {
    mockFeatureFlags({ traceGraphEnabled: false });

    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-001');

    expect(await screen.findByText('Trace Answer customer policy question')).toBeInTheDocument();
    expect(screen.queryByText('Graph')).not.toBeInTheDocument();
    expect(screen.getByText('Tree')).toBeInTheDocument();
    expect(screen.getByText('List')).toBeInTheDocument();
  });

  it('keeps the detail shell useful for a session with one trace', async () => {
    renderSessionDetail('session-agent-run-002');

    expect(await screen.findByText('Session session-agent-run-002')).toBeInTheDocument();
    expect(screen.queryByText('1 trace')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('session-summary-header')).getByText('3')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /View trace/ })).not.toBeInTheDocument();
    const trajectory = screen.getByRole('navigation', { name: 'Trace trajectory' });
    expect(within(trajectory).getByText('Session')).toBeInTheDocument();
    expect(
      (await within(trajectory).findAllByText('Retrieve deployment troubleshooting steps')).length
    ).toBeGreaterThan(0);
  });

  it('restores a span deep link without first loading the root span detail', async () => {
    const detailSpanIds: string[] = [];
    const spanListRequests: URL[] = [];
    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        const url = new URL(request.url);
        spanListRequests.push(url);
        const sessionId = url.searchParams.get('filter[session_id]');
        const traceId = url.searchParams.get('filter[trace_id]');
        const data = mockSpansPage.data.filter(
          (span) => span.session_id === sessionId && (!traceId || span.trace_id === traceId)
        );
        return HttpResponse.json({
          ...mockSpansPage,
          data,
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: data.length,
            total_results: data.length,
          },
        });
      }),
      http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), ({ params }) => {
        const spanId = String(params['spanId']);
        detailSpanIds.push(spanId);
        const span = mockSpanById(spanId);
        return span ? HttpResponse.json(span) : new HttpResponse(null, { status: 404 });
      })
    );

    renderSessionDetail(
      'session-agent-run-001',
      '?traceId=trace-agent-run-001&spanId=span-llm-001'
    );

    expect((await screen.findAllByText('Generate final response')).length).toBeGreaterThan(0);
    expect(screen.getByText('Trace Answer customer policy question')).toBeInTheDocument();
    await waitFor(() => expect(detailSpanIds).toContain('span-llm-001'));
    expect(spanListRequests).toHaveLength(2);
    expect(spanListRequests.every((url) => url.searchParams.get('mode') === 'summary')).toBe(true);
    expect(
      spanListRequests.some(
        (url) => url.searchParams.get('filter[trace_id]') === 'trace-agent-run-001'
      )
    ).toBe(true);
    expect(detailSpanIds).not.toContain('span-root-001');
  });

  it('preserves a directly linked span outside the loaded summary page', async () => {
    mockFeatureFlags({ traceGraphEnabled: true });
    const user = userEvent.setup();
    let resolveOutsidePageSpan!: () => void;
    const outsidePageSpanGate = new Promise<void>((resolve) => {
      resolveOutsidePageSpan = resolve;
    });
    const outsidePageSpan = {
      ...mockSpanById('span-llm-001')!,
      span_id: 'span-outside-page-001',
      parent_span_id: 'span-missing-from-page',
      name: 'Outside page span',
    };

    server.use(
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), ({ request }) => {
        const sessionId = new URL(request.url).searchParams.get('filter[session_id]');
        const data = mockSpansPage.data.filter(
          (span) => span.session_id === sessionId && span.span_id === 'span-root-001'
        );
        return HttpResponse.json({
          ...mockSpansPage,
          data,
          pagination: { ...mockSpansPage.pagination, total_results: 1001, total_pages: 2 },
        });
      }),
      http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), async ({ params }) => {
        if (params['spanId'] === outsidePageSpan.span_id) {
          await outsidePageSpanGate;
          return HttpResponse.json(outsidePageSpan);
        }
        const span = mockSpanById(String(params['spanId']));
        return span ? HttpResponse.json(span) : new HttpResponse(null, { status: 404 });
      })
    );

    renderSessionDetail(
      'session-agent-run-001',
      '?traceId=trace-agent-run-001&spanId=span-outside-page-001'
    );

    await user.click(await screen.findByText('List'));
    expect(await screen.findByLabelText('Loading linked span')).toBeInTheDocument();
    resolveOutsidePageSpan();
    expect((await screen.findAllByText('Outside page span')).length).toBeGreaterThan(0);

    await user.click(screen.getByText('Graph'));
    await user.click(screen.getByText('Grouped'));
    expect(screen.getByText('2 groups from 2 spans')).toBeInTheDocument();
    expect(screen.getAllByText('Outside page span')).not.toHaveLength(0);
  });

  it('renders received spans while trace details are still arriving', async () => {
    const traceId = 'trace-still-arriving';
    const receivedSpan = {
      ...mockSpanById('span-llm-001')!,
      span_id: 'span-received-before-trace',
      parent_span_id: 'span-root-not-received',
      trace_id: traceId,
      name: 'Generate response from received activity',
    };

    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockTracesPage,
          data: [],
          pagination: {
            ...mockTracesPage.pagination,
            current_page_size: 0,
            total_results: 0,
          },
        })
      ),
      http.get(mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'), ({ params }) =>
        params['traceId'] === traceId
          ? new HttpResponse(null, { status: 404 })
          : new HttpResponse(null, { status: 500 })
      ),
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockSpansPage,
          data: [receivedSpan],
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: 1,
            total_results: 1,
          },
        })
      )
    );

    renderSessionDetail('session-agent-run-001', `?traceId=${traceId}`);

    expect(
      await screen.findByText(
        'Trace details are still arriving. The activity received so far is shown below.'
      )
    ).toBeInTheDocument();
    expect(screen.queryByText('Trace Not Found')).not.toBeInTheDocument();
    expect(
      (await screen.findAllByText('Generate response from received activity')).length
    ).toBeGreaterThan(0);
  });

  it('shows not found when neither trace details nor matching spans exist', async () => {
    const traceId = 'trace-never-received';
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockTracesPage,
          data: [],
          pagination: {
            ...mockTracesPage.pagination,
            current_page_size: 0,
            total_results: 0,
          },
        })
      ),
      http.get(
        mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'),
        () => new HttpResponse(null, { status: 404 })
      ),
      http.get(mockApiUrl(getListSpansQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockSpansPage,
          data: [],
          pagination: {
            ...mockSpansPage.pagination,
            current_page_size: 0,
            total_results: 0,
          },
        })
      )
    );

    renderSessionDetail('session-agent-run-001', `?traceId=${traceId}`);

    expect(await screen.findByText('Trace Not Found')).toBeInTheDocument();
    expect(screen.queryByText(/Trace details are still arriving/)).not.toBeInTheDocument();
  });

  it('shows an error when trace details are missing and session spans fail to load', async () => {
    const traceId = 'trace-with-failed-activity-request';
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({
          ...mockTracesPage,
          data: [],
          pagination: {
            ...mockTracesPage.pagination,
            current_page_size: 0,
            total_results: 0,
          },
        })
      ),
      http.get(
        mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'),
        () => new HttpResponse(null, { status: 404 })
      ),
      http.get(
        mockApiUrl(getListSpansQueryKey, ':workspace'),
        () => new HttpResponse(null, { status: 500 })
      )
    );

    renderSessionDetail('session-agent-run-001', `?traceId=${traceId}`);

    expect(await screen.findByText('Error loading trace activity')).toBeInTheDocument();
    expect(screen.queryByText('Trace Not Found')).not.toBeInTheDocument();
  });

  it('rejects a trace deep link that belongs to another session', async () => {
    renderSessionDetail('session-agent-run-001', '?traceId=trace-agent-run-002');

    expect(await screen.findByText('Trace Not Found')).toBeInTheDocument();
    expect(screen.getByText(/does not belong to this session/)).toBeInTheDocument();
  });
});
