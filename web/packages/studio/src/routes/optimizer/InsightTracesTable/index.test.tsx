// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import type { Trace } from '@nemo/sdk/generated/platform/schema';
import { getListTracesQueryKey } from '@nemo/sdk/generated/platform/traces';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { InsightTracesTable } from '@studio/routes/optimizer/InsightTracesTable';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const makeTrace = (sequence: number): Trace => ({
  id: `trace-${String(sequence).padStart(2, '0')}`,
  session_id: `session-${sequence}`,
  workspace: DEFAULT_WORKSPACE,
  name: `Trace ${String(sequence).padStart(2, '0')}`,
  started_at: `2026-07-20T12:${String(sequence).padStart(2, '0')}:00Z`,
  status: 'success',
});

const installTraceHandler = ({
  traces,
  missingIds = [],
}: {
  traces: Trace[];
  missingIds?: string[];
}) => {
  const requests: URL[] = [];
  const tracesById = new Map(traces.map((trace) => [trace.id, trace]));

  server.use(
    http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), ({ request }) => {
      const url = new URL(request.url);
      requests.push(url);
      const requestedIds = url.searchParams.getAll('filter[id][$in]');
      const data = requestedIds
        .filter((id) => !missingIds.includes(id))
        .flatMap((id) => {
          const trace = tracesById.get(id);
          return trace ? [trace] : [];
        })
        .reverse();

      return HttpResponse.json({ data });
    })
  );

  return requests;
};

describe('InsightTracesTable', () => {
  it('requests and displays only the current page in reference order using preview mode', async () => {
    const user = userEvent.setup();
    const traces = Array.from({ length: 11 }, (_, index) => makeTrace(index + 1));
    const traceIds = traces.map((trace) => trace.id);
    const missingId = traces[9].id;
    const requests = installTraceHandler({ traces, missingIds: [missingId] });

    renderRoute(<InsightTracesTable workspace={DEFAULT_WORKSPACE} traceIds={traceIds} />, {
      history: '/optimizer?page_size=10',
    });

    expect(screen.queryByText("10 of 10 traces couldn't be loaded.")).not.toBeInTheDocument();

    await screen.findByText('Trace 01');
    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].searchParams.get('mode')).toBe('preview');
    expect(requests[0].searchParams.getAll('filter[id][$in]')).toEqual(traceIds.slice(0, 10));
    expect(screen.getByText("1 of 10 traces couldn't be loaded.")).toBeInTheDocument();
    expect(screen.queryByText('Trace 10')).not.toBeInTheDocument();
    expect(screen.queryByText('Trace 11')).not.toBeInTheDocument();
    const rows = screen.getAllByRole('row');
    expect(rows[1]).toHaveTextContent('Trace 01');
    expect(rows[2]).toHaveTextContent('Trace 02');
    expect(screen.getByText('1-10 of 11 items')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /next page/i }));

    expect(await screen.findByText('Trace 11')).toBeInTheDocument();
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1].searchParams.getAll('filter[id][$in]')).toEqual(['trace-11']);
    expect(screen.queryByText('Trace 01')).not.toBeInTheDocument();
    expect(screen.getByText('11-11 of 11 items')).toBeInTheDocument();
  });

  it('shows an error instead of an empty state when the list request fails', async () => {
    server.use(
      http.get(mockApiUrl(getListTracesQueryKey, ':workspace'), () =>
        HttpResponse.json({ detail: 'Could not load traces' }, { status: 500 })
      )
    );

    renderRoute(<InsightTracesTable workspace={DEFAULT_WORKSPACE} traceIds={[makeTrace(1).id]} />);

    expect(await screen.findByText('Error')).toBeInTheDocument();
    expect(
      screen.queryByText(ENTITY_EMPTY_STATES.insightTraces.subheading)
    ).not.toBeInTheDocument();
  });
});
