// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Checkbox, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { LineChartSkeleton } from '@studio/components/charts/LineChartSkeleton';
import type { RunSummary } from '@studio/routes/agents/AgentMonitorRoute/telemetry';
import { FC, type ReactElement, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface Props {
  runs: RunSummary[];
  isPending?: boolean;
  height?: number;
}

const MAX_SPACERS_PROPORTIONAL = 5;
const MAX_SPACERS_COLLAPSED = 2;
const BREAK_RATIO = 4;
const BRUSH_HEIGHT = 20;
/** How many data slots are visible in the brush window initially. */
const BRUSH_VISIBLE_SLOTS = 20;

interface ChartPoint {
  id: string;
  realTime: number | null;
  prompt: number | null;
  completion: number | null;
}

interface ProcessedData {
  points: ChartPoint[];
  gapIds: string[];
}

const medianOf = (values: number[]): number => {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[m] : (s[m - 1] + s[m]) / 2;
};

type SpacingMode = 'proportional' | 'collapse';

const processRuns = (runs: RunSummary[], spacingMode: SpacingMode): ProcessedData => {
  const sorted = [...runs].sort((a, b) => a.startedAt.getTime() - b.startedAt.getTime());
  if (sorted.length === 0) return { points: [], gapIds: [] };

  const makeRun = (i: number): ChartPoint => ({
    id: `run-${i}`,
    realTime: sorted[i].startedAt.getTime(),
    prompt: sorted[i].promptTokens,
    completion: sorted[i].completionTokens,
  });

  if (sorted.length === 1) return { points: [makeRun(0)], gapIds: [] };

  const times = sorted.map((r) => r.startedAt.getTime());
  const gaps = times.slice(1).map((t, i) => t - times[i]);
  const med = medianOf(gaps);

  const points: ChartPoint[] = [];
  const gapIds: string[] = [];

  for (let i = 0; i < sorted.length; i++) {
    if (i > 0) {
      const gap = gaps[i - 1];
      let spacerCount = 0;
      let needsBreakMarker = false;

      if (med > 0) {
        const proportional = Math.max(0, Math.round(gap / med) - 1);
        if (spacingMode === 'collapse') {
          spacerCount = Math.min(MAX_SPACERS_COLLAPSED, proportional);
          needsBreakMarker = gap > BREAK_RATIO * med;
        } else {
          spacerCount = Math.min(MAX_SPACERS_PROPORTIONAL, proportional);
        }
      }

      for (let s = 0; s < spacerCount; s++) {
        const sid = `gap-${i}-${s}`;
        if (s === 0 && needsBreakMarker) gapIds.push(sid);
        points.push({ id: sid, realTime: null, prompt: null, completion: null });
      }
    }
    points.push(makeRun(i));
  }

  return { points, gapIds };
};

const formatLabel = (ts: number): string =>
  new Date(ts).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

const TICK_STYLE = { fontSize: 11, fill: 'var(--text-color-base)' } as const;

const brushTraveller = (
  <rect rx={3} fill="var(--text-color-brand)" stroke="none" />
) as ReactElement<SVGElement>;

const BREAK_COLOR = 'var(--text-color-subtle, #888)';
const chartMargin = { left: 16, right: 16, bottom: 56 };

export const TokenUsageChart: FC<Props> = ({ runs, isPending, height = 320 }) => {
  const [collapseGaps, setCollapseGaps] = useState(false);

  const { points, gapIds } = useMemo(
    () => processRuns(runs, collapseGaps ? 'collapse' : 'proportional'),
    [runs, collapseGaps]
  );

  const idToRealTime = useMemo(
    () =>
      new Map(points.filter((p) => p.realTime != null).map((p) => [p.id, p.realTime as number])),
    [points]
  );

  const brushStart = Math.max(0, points.length - BRUSH_VISIBLE_SLOTS);
  const brushEnd = points.length - 1;
  const totalHeight = height + BRUSH_HEIGHT + 16;

  if (isPending) {
    return <LineChartSkeleton />;
  }

  const chart = (
    <BarChart data={points} barSize={16} margin={chartMargin}>
      <CartesianGrid strokeDasharray="3 3" vertical={false} />
      <XAxis
        dataKey="id"
        tickFormatter={(id: string) => {
          const rt = idToRealTime.get(id);
          return rt != null ? formatLabel(rt) : '';
        }}
        tick={TICK_STYLE}
        angle={-40}
        textAnchor="end"
        height={72}
        interval={0}
      />
      <YAxis tick={TICK_STYLE} width={56} />
      <Tooltip
        labelFormatter={(id: string) => {
          const rt = idToRealTime.get(id);
          return rt != null ? formatLabel(rt) : '';
        }}
        cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
        contentStyle={{
          fontSize: 12,
          backgroundColor: 'var(--background-color-component-tooltip)',
          borderColor: 'var(--border-color-base)',
          color: 'var(--text-color-base)',
        }}
        labelStyle={{ color: 'var(--text-color-base)' }}
        itemStyle={{ color: 'var(--text-color-base)' }}
      />
      {gapIds.map((gapId) => (
        <ReferenceLine
          key={gapId}
          x={gapId}
          stroke={BREAK_COLOR}
          strokeDasharray="4 2"
          label={{ value: '⋯', position: 'top', fill: BREAK_COLOR, fontSize: 16 }}
        />
      ))}
      <Bar dataKey="prompt" name="Prompt" stackId="tokens" fill="var(--text-color-brand)" />
      <Bar
        dataKey="completion"
        name="Completion"
        stackId="tokens"
        fill="var(--text-color-accent-purple)"
      />
      <Brush
        dataKey="id"
        height={BRUSH_HEIGHT}
        startIndex={brushStart}
        endIndex={brushEnd}
        tickFormatter={(id: string) => {
          const rt = idToRealTime.get(id);
          return rt != null ? formatLabel(rt) : '';
        }}
        traveller={brushTraveller}
        stroke="var(--text-color-subtle, #666)"
        fill="rgba(128, 128, 128, 0.1)"
        travellerWidth={6}
      />
    </BarChart>
  );

  return (
    <Card>
      <Stack gap="density-md" padding="density-xl">
        <Flex justify="between" align="center">
          <Stack gap="density-xs">
            <Text kind="title/sm">Token usage over time</Text>
            <Text className="text-secondary" kind="body/regular/sm">
              Prompt and completion tokens per run
            </Text>
          </Stack>
          <Checkbox
            checked={collapseGaps}
            onCheckedChange={(checked) => setCollapseGaps(checked === true)}
            slotLabel="Collapse"
          />
        </Flex>
        {points.length > 0 ? (
          <Stack gap="density-xs">
            <Flex gap="density-md" align="center">
              {(
                [
                  { label: 'Prompt', swatchClass: 'bg-brand' },
                  { label: 'Completion', swatchClass: 'bg-accent-purple-strong' },
                ] as const
              ).map(({ label, swatchClass }) => (
                <Flex key={label} gap="density-xs" align="center">
                  <div className={`h-3 w-3 shrink-0 rounded-sm ${swatchClass}`} />
                  <Text kind="body/regular/sm">{label}</Text>
                </Flex>
              ))}
            </Flex>
            <ResponsiveContainer width="100%" height={totalHeight}>
              {chart}
            </ResponsiveContainer>
          </Stack>
        ) : (
          <Text className="text-secondary" kind="body/regular/sm">
            No telemetry runs available — invoke an agent to populate this view.
          </Text>
        )}
      </Stack>
    </Card>
  );
};
