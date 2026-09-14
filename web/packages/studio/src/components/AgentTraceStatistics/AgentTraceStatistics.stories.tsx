// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import {
  AgentTraceStatistics,
  type AgentTraceStatisticsProps,
} from '@studio/components/AgentTraceStatistics/index';
import type {
  TraceStatisticsBucket,
  TraceStatisticsRange,
  TraceStatisticsSummary,
} from '@studio/components/AgentTraceStatistics/types';
import { type FC, useState } from 'react';

const ANCHOR = new Date('2026-07-01T00:00:00Z').getTime();
const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const WEEK_MS = 7 * DAY_MS;

/** Deterministic LCG — fixtures must not change between story renders. */
const makeRandom = (seed: number): (() => number) => {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
};

interface FixtureOptions {
  buckets: number;
  bucketMs: number;
  runsPerBucket?: number;
  seed?: number;
}

interface Fixture {
  summary: TraceStatisticsSummary;
  buckets: TraceStatisticsBucket[];
}

/**
 * Stands in for Intake's bucketed rollup: token means swing on a slow sine so the chart shows the
 * peaks and troughs of a real workload, with latency and cost derived from tokens plus jitter. The
 * summary is the same numbers collapsed, the way `bucket=total` would return them.
 */
const makeFixture = ({
  buckets,
  bucketMs,
  runsPerBucket = 6,
  seed = 42,
}: FixtureOptions): Fixture => {
  const random = makeRandom(seed);
  const points: TraceStatisticsBucket[] = [];
  const start = ANCHOR - (buckets - 1) * bucketMs;

  for (let bucket = 0; bucket < buckets; bucket++) {
    const wave = Math.sin(bucket / 1.7) * 0.5 + Math.sin(bucket / 5.3) * 0.5;
    const tokens = Math.max(120, Math.round((2200 + wave * 1300) * (0.85 + random() * 0.3)));
    points.push({
      timestamp: start + bucket * bucketMs,
      tokens,
      latencyMs: Math.round(tokens * (0.045 + random() * 0.02) * 1000) / 1000,
      costUsd: tokens * 0.0000032 * (0.9 + random() * 0.2),
    });
  }

  const mean = (select: (point: TraceStatisticsBucket) => number | null): number =>
    points.reduce((total, point) => total + (select(point) ?? 0), 0) / points.length;

  return {
    summary: {
      totalTraces: points.length * runsPerBucket,
      avgLatencyMs: mean((point) => point.latencyMs),
      avgTokensPerRun: mean((point) => point.tokens),
      avgCostUsd: mean((point) => point.costUsd),
    },
    buckets: points,
  };
};

const MONTH = makeFixture({ buckets: 31, bucketMs: DAY_MS });
const WEEK = makeFixture({ buckets: 7, bucketMs: DAY_MS, seed: 7 });
const DAY = makeFixture({ buckets: 24, bucketMs: HOUR_MS, runsPerBucket: 3, seed: 11 });
const MAX = makeFixture({ buckets: 13, bucketMs: WEEK_MS, runsPerBucket: 30, seed: 23 });

const FIXTURE_BY_RANGE: Record<TraceStatisticsRange, Fixture> = {
  day: DAY,
  week: WEEK,
  month: MONTH,
  max: MAX,
};

const meta: Meta<typeof AgentTraceStatistics> = {
  component: AgentTraceStatistics,
  title: 'Studio/AgentTraceStatistics',
  args: {
    range: 'month',
    summary: MONTH.summary,
    buckets: MONTH.buckets,
    onRangeChange: () => {},
    onViewTraces: () => {},
  },
  argTypes: {
    range: { control: 'select', options: ['day', 'week', 'month', 'max'] },
    chartHeight: { control: { type: 'range', min: 160, max: 600, step: 20 } },
  },
  decorators: [
    (Story) => (
      <div className="w-full max-w-[1200px] p-6">
        <Story />
      </div>
    ),
  ],
};

export default meta;

type Story = StoryObj<typeof AgentTraceStatistics>;

/** Stands in for the caller's refetch: changing the range swaps which fixture is passed in. */
const RangeAwareStatistics: FC<AgentTraceStatisticsProps> = (props) => {
  const [range, setRange] = useState<TraceStatisticsRange>(props.range);
  const fixture = FIXTURE_BY_RANGE[range];
  return (
    <AgentTraceStatistics
      {...props}
      range={range}
      summary={fixture.summary}
      buckets={fixture.buckets}
      onRangeChange={setRange}
    />
  );
};

export const Default: Story = {
  render: (args) => <RangeAwareStatistics {...args} />,
};

/** Hourly buckets — the tick formatter switches from dates to hours. */
export const DayRange: Story = {
  args: { range: 'day', summary: DAY.summary, buckets: DAY.buckets },
};

export const Loading: Story = {
  args: { isPending: true, summary: null, buckets: [] },
};

/**
 * First run: the tiles and chart give way to instructions, since four zeros and a blank grid tell
 * the user nothing about how to get data.
 */
export const Empty: Story = {
  args: {
    summary: null,
    buckets: [],
    onRunAgent: () => {},
    onLearnMore: () => {},
  },
};

/** Buckets missing cost or token rollups still plot the series they do have. */
export const PartialRollups: Story = {
  args: {
    buckets: MONTH.buckets.map((bucket, index) =>
      index % 3 === 0 ? { ...bucket, costUsd: null, tokens: null } : bucket
    ),
  },
};

/** No `onViewTraces` hides the action — for surfaces that are already the traces list. */
export const WithoutViewTracesAction: Story = {
  args: { onViewTraces: undefined },
};
