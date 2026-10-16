// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanKind, SpanStatus } from '@nemo/sdk/generated/platform/schema';
import { getGetSpanQueryKey } from '@nemo/sdk/generated/platform/spans';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { TestCaseTabs } from '@studio/routes/EvaluationSessionDetailRoute/TestCaseTabs';
import { render, screen } from '@studio/tests/util/render';
import { buildSpanTree, type SessionTrajectory } from '@studio/util/intakeTelemetry';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const span = (overrides: Partial<Span> & Pick<Span, 'span_id' | 'kind'>): Span => ({
  session_id: 'session',
  workspace: 'default',
  source: 'test',
  trace_id: 'trace',
  started_at: '2026-10-16T00:00:00Z',
  ingested_at: '2026-10-16T00:00:00Z',
  status: SpanStatus.success,
  ...overrides,
});

const trajectoryOf = (spans: Span[]): SessionTrajectory[] => [
  {
    trace: {
      id: 'trace',
      session_id: 'session',
      workspace: 'default',
      started_at: '2026-10-16T00:00:00Z',
      status: SpanStatus.success,
    },
    spans,
    spanTree: buildSpanTree(spans),
  },
];

const mockReply = (output: string | undefined) =>
  server.use(
    http.get(mockApiUrl(getGetSpanQueryKey, ':workspace', ':spanId'), ({ params }) =>
      HttpResponse.json(span({ span_id: String(params['spanId']), kind: SpanKind.AGENT, output }))
    )
  );

describe('TestCaseTabs', () => {
  it('shows the agent reply and each tool it called', async () => {
    mockReply('The plan is 61% above demand.');
    const spans = [
      span({ span_id: 'agent', kind: SpanKind.AGENT }),
      span({
        span_id: 'tool-1',
        kind: SpanKind.TOOL,
        parent_span_id: 'agent',
        tool_name: 'get_demand',
      }),
      span({
        span_id: 'tool-2',
        kind: SpanKind.TOOL,
        parent_span_id: 'agent',
        name: 'score_schedule',
      }),
    ];

    const user = userEvent.setup();
    render(
      <TestCaseTabs
        workspace="default"
        trajectories={trajectoryOf(spans)}
        traceView={<div>Span tree</div>}
      />
    );

    expect(await screen.findByText('The plan is 61% above demand.')).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Trace' }));
    expect(await screen.findByText('Tool calls (2)')).toBeInTheDocument();
    expect(screen.getByText('Span tree')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /score_schedule/ })).toBeInTheDocument();

    const getDemand = screen.getByRole('button', { name: /get_demand/ });
    await user.click(getDemand);
    expect(getDemand).toHaveClass('nv-button--kind-primary');
  });

  it('says when no reply or tool calls were recorded', async () => {
    mockReply(undefined);
    const user = userEvent.setup();

    render(
      <TestCaseTabs
        workspace="default"
        trajectories={trajectoryOf([span({ span_id: 'wrapper', kind: SpanKind.CHAIN })])}
        traceView={null}
      />
    );

    expect(await screen.findByText('No reply was recorded for this test.')).toBeInTheDocument();
    await user.click(screen.getByRole('tab', { name: 'Trace' }));
    expect(
      await screen.findByText('No tool calls were recorded for this test.')
    ).toBeInTheDocument();
  });
});
