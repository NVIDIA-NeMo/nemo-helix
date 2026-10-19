// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentTraceStatistics } from '@studio/components/AgentTraceStatistics/index';
import type {
  TraceStatisticsBucket,
  TraceStatisticsSummary,
} from '@studio/components/AgentTraceStatistics/types';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { AxiosError, AxiosHeaders } from 'axios';

const SUMMARY: TraceStatisticsSummary = {
  totalTraces: 2,
  avgLatencyMs: 2500,
  avgTokensPerRun: 1500,
  avgCostUsd: 0.03,
};

const BUCKETS: TraceStatisticsBucket[] = [
  {
    timestamp: new Date('2026-07-01T00:00:00Z').getTime(),
    costUsd: 0.02,
    tokens: 1000,
    latencyMs: 2000,
  },
  {
    timestamp: new Date('2026-07-02T00:00:00Z').getTime(),
    costUsd: 0.04,
    tokens: 2000,
    latencyMs: 3000,
  },
];

const noop = () => {};

describe('AgentTraceStatistics', () => {
  it('renders the summary tiles from the rollup it is given', () => {
    renderRoute(
      <AgentTraceStatistics
        summary={SUMMARY}
        buckets={BUCKETS}
        range="week"
        onRangeChange={noop}
        onViewTraces={noop}
      />
    );

    expect(screen.getByText('Trace statistics')).toBeInTheDocument();
    expect(screen.getByText('Total traces')).toBeInTheDocument();
    expect(screen.getByText('1,500')).toBeInTheDocument();
    expect(screen.getByText('2,500 ms')).toBeInTheDocument();
    expect(screen.getByText('$0.03')).toBeInTheDocument();
  });

  it('shows the selected range as its option label, not the raw value', () => {
    renderRoute(
      <AgentTraceStatistics summary={SUMMARY} buckets={BUCKETS} range="week" onRangeChange={noop} />
    );

    expect(screen.getByRole('combobox', { name: 'Statistics range' })).toHaveTextContent('Week');
  });

  it('replaces the tiles and chart with actionable guidance when there are no traces', () => {
    renderRoute(
      <AgentTraceStatistics
        summary={null}
        buckets={[]}
        range="week"
        onRangeChange={noop}
        onViewTraces={noop}
        onRunAgent={noop}
        onLearnMore={noop}
      />
    );

    expect(screen.getByText('No traces')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /run the agent/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /set up tracing/i })).toBeInTheDocument();
    expect(screen.queryByText('Total traces')).not.toBeInTheDocument();
    // Nothing to navigate to when the range is empty.
    expect(screen.queryByRole('button', { name: /view traces/i })).not.toBeInTheDocument();
  });

  it('shows the tiles while the rollup is still loading rather than the empty state', () => {
    renderRoute(
      <AgentTraceStatistics
        summary={null}
        buckets={[]}
        range="week"
        onRangeChange={noop}
        isPending
      />
    );

    expect(screen.queryByText('No traces')).not.toBeInTheDocument();
  });

  it('offers to widen a narrow empty range and reports the change', async () => {
    const onRangeChange = vi.fn();
    renderRoute(
      <AgentTraceStatistics
        summary={null}
        buckets={[]}
        range="week"
        onRangeChange={onRangeChange}
      />
    );

    await userEvent.click(screen.getByRole('button', { name: /look back further/i }));
    expect(onRangeChange).toHaveBeenCalledWith('max');
  });

  it('does not offer to widen when already on the longest range', () => {
    renderRoute(
      <AgentTraceStatistics summary={null} buckets={[]} range="max" onRangeChange={noop} />
    );

    expect(screen.queryByRole('button', { name: /look back further/i })).not.toBeInTheDocument();
  });

  it('forwards a network-level failure message rather than a hardcoded one', () => {
    renderRoute(
      <AgentTraceStatistics
        summary={SUMMARY}
        buckets={BUCKETS}
        range="week"
        onRangeChange={noop}
        onViewTraces={noop}
        error={new AxiosError('Network Error', 'ERR_NETWORK')}
      />
    );

    expect(screen.getByText('Trace statistics')).toBeInTheDocument();
    expect(screen.getByText('Trace statistics are unavailable')).toBeInTheDocument();
    expect(screen.getByText(/\[ERR_NETWORK\] Network Error/)).toBeInTheDocument();
    expect(screen.queryByText('Total traces')).not.toBeInTheDocument();
    expect(screen.queryByText('No traces')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /view traces/i })).not.toBeInTheDocument();
  });

  it('falls back to the generic ClickHouse message when the error carries nothing useful', () => {
    renderRoute(
      <AgentTraceStatistics
        summary={SUMMARY}
        buckets={BUCKETS}
        range="week"
        onRangeChange={noop}
        error="not an Error instance"
      />
    );

    expect(screen.getByText(/trace store couldn't be reached/i)).toBeInTheDocument();
    expect(screen.getByText(/ClickHouse is down/i)).toBeInTheDocument();
  });

  it("prefers the backend's own detail message when it responded with one", () => {
    const error = new AxiosError('Request failed', 'ERR_BAD_RESPONSE', undefined, undefined, {
      status: 503,
      statusText: 'Service Unavailable',
      data: { detail: 'ClickHouse storage is inaccessible. Check that ClickHouse is running.' },
      headers: {},
      config: {
        headers: new AxiosHeaders(),
      },
    });

    renderRoute(
      <AgentTraceStatistics
        summary={SUMMARY}
        buckets={BUCKETS}
        range="week"
        onRangeChange={noop}
        error={error}
      />
    );

    expect(
      screen.getByText('ClickHouse storage is inaccessible. Check that ClickHouse is running.')
    ).toBeInTheDocument();
  });
});
