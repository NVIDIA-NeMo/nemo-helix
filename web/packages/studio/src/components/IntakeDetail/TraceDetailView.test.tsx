// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { Trace } from '@nemo/sdk/generated/platform/schema';
import { getGetTraceQueryKey } from '@nemo/sdk/generated/platform/traces';
import { TraceDetailView } from '@studio/components/IntakeDetail/TraceDetailView';
import { mockTraceById } from '@studio/mocks/intake/telemetry';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import { useIsFetching } from '@tanstack/react-query';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { Link, useParams } from 'react-router';

const sharedTraceId = 'reused-trace-id';
const traces: Trace[] = ['session-a', 'session-b'].map((sessionId) => ({
  ...mockTraceById('trace-agent-run-001')!,
  id: sharedTraceId,
  session_id: sessionId,
  name: `Detailed trace for ${sessionId}`,
}));

const TracePage = () => {
  const { sessionId = '' } = useParams();
  const pendingRequests = useIsFetching();
  const summary = traces.find((trace) => trace.session_id === sessionId);
  return (
    <>
      <output>{pendingRequests === 0 ? 'Details loaded' : 'Details loading'}</output>
      <Link to="/sessions/session-a">Session A</Link>
      <Link to="/sessions/session-b">Session B</Link>
      <TraceDetailView
        workspace={DEFAULT_WORKSPACE}
        traceId={sharedTraceId}
        sessionId={sessionId}
        traceSummary={summary ? { ...summary, name: 'Summary while loading' } : undefined}
        traceSummaryStatus="resolved"
        parentBreadcrumbs={[]}
      >
        {(trace) => <output data-testid="selected-trace">{trace.name}</output>}
      </TraceDetailView>
    </>
  );
};

const renderTracePage = (sessionId = 'session-a') =>
  renderRoute(undefined, {
    history: `/sessions/${sessionId}`,
    routes: [{ path: '/sessions/:sessionId', element: <TracePage /> }],
    testProviderOptions: {
      queryClientConfig: { defaultOptions: { queries: { retry: false, staleTime: Infinity } } },
    },
  });

describe('TraceDetailView session identity', () => {
  it('fetches and caches colliding trace IDs separately while navigating sessions', async () => {
    const requests: string[] = [];
    server.use(
      http.get(mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'), ({ request }) => {
        const sessionId = new URL(request.url).searchParams.get('session_id');
        requests.push(sessionId ?? 'unscoped');
        const trace = traces.find((candidate) => candidate.session_id === sessionId);
        return trace ? HttpResponse.json(trace) : new HttpResponse(null, { status: 409 });
      })
    );
    const user = userEvent.setup();
    renderTracePage();

    await waitFor(() =>
      expect(screen.getByTestId('selected-trace')).toHaveTextContent('Detailed trace for session-a')
    );
    await user.click(screen.getByRole('link', { name: 'Session B' }));
    await waitFor(() =>
      expect(screen.getByTestId('selected-trace')).toHaveTextContent('Detailed trace for session-b')
    );
    const completedRequests = requests.length;
    await user.click(screen.getByRole('link', { name: 'Session A' }));
    expect(screen.getByTestId('selected-trace')).toHaveTextContent('Detailed trace for session-a');
    expect(new Set(requests)).toEqual(new Set(['session-a', 'session-b']));
    expect(requests).toHaveLength(completedRequests);
    expect(screen.queryByText('Trace Not Found')).not.toBeInTheDocument();
  });

  it('explains an ambiguous lookup when no session summary is available', async () => {
    server.use(
      http.get(
        mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'),
        () => new HttpResponse(null, { status: 409 })
      )
    );
    renderTracePage('missing-session');

    expect(await screen.findByText('Trace ID is ambiguous')).toBeInTheDocument();
    expect(screen.getByText(/belongs to multiple sessions/)).toBeInTheDocument();
    expect(screen.queryByTestId('selected-trace')).not.toBeInTheDocument();
  });

  it('does not replace a session summary with detail from a different session', async () => {
    let responded = false;
    server.use(
      http.get(mockApiUrl(getGetTraceQueryKey, ':workspace', ':traceId'), () => {
        responded = true;
        return HttpResponse.json(traces[1]);
      })
    );
    renderTracePage();

    await waitFor(() => expect(responded).toBe(true));
    expect(await screen.findByText('Details loaded')).toBeInTheDocument();
    expect(screen.getByTestId('selected-trace')).toHaveTextContent('Summary while loading');
    expect(screen.queryByText('Trace Not Found')).not.toBeInTheDocument();
    expect(screen.queryByText('Detailed trace for session-b')).not.toBeInTheDocument();
  });
});
