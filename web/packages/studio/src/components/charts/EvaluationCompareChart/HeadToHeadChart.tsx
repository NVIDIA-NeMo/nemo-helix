// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLegend } from '@nemo/common/src/components/charts/ChartLegend';
import {
  ChartTooltipRow,
  ChartTooltipSurface,
} from '@nemo/common/src/components/charts/ChartTooltip';
import { TICK_STYLE } from '@nemo/common/src/components/charts/frame';
import { AXIS_COLOR } from '@nemo/common/src/components/charts/tokens';
import { Stack } from '@nvidia/foundations-react-core';
import type { HeadToHead } from '@studio/components/charts/EvaluationCompareChart/utils';
import type { FC } from 'react';
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  type TooltipProps,
  XAxis,
  YAxis,
} from 'recharts';

const ROW_HEIGHT = 36;
const CATEGORY_WIDTH = 140;

const OUTCOMES = [
  { key: 'better', label: 'Better', color: 'var(--text-color-feedback-success)' },
  { key: 'same', label: 'Same', color: 'var(--text-color-accent-gray)' },
  { key: 'worse', label: 'Worse', color: 'var(--text-color-feedback-danger)' },
] as const satisfies readonly { key: keyof HeadToHead; label: string; color: string }[];

const LEGEND_ITEMS = OUTCOMES.map(({ key, label, color }) => ({ id: key, label, color }));

export interface HeadToHeadRow extends HeadToHead {
  readonly name: string;
}

const total = (row: HeadToHead): number => row.better + row.same + row.worse;

const HeadToHeadTooltip: FC<TooltipProps<number, string>> = ({ active, payload }) => {
  const row = payload?.[0]?.payload as HeadToHeadRow | undefined;
  if (!active || !row) return null;
  const count = total(row);
  return (
    <ChartTooltipSurface label={`${row.name} vs baseline`}>
      {OUTCOMES.map(({ key, label, color }) => (
        <ChartTooltipRow
          key={key}
          color={color}
          label={label}
          value={`${row[key]} of ${count} test cases`}
        />
      ))}
    </ChartTooltipSurface>
  );
};

const truncate = (label: string): string => (label.length > 20 ? `${label.slice(0, 19)}…` : label);

export const HeadToHeadChart: FC<{ rows: readonly HeadToHeadRow[] }> = ({ rows }) => (
  <Stack gap="density-sm">
    <ChartLegend items={LEGEND_ITEMS} interactive={false} justify="end" />
    <ResponsiveContainer width="100%" height={rows.length * ROW_HEIGHT + 32}>
      <BarChart
        data={[...rows]}
        layout="vertical"
        stackOffset="expand"
        margin={{ top: 0, right: 24, bottom: 0, left: 0 }}
        barSize={20}
      >
        <XAxis
          type="number"
          tick={TICK_STYLE}
          tickLine={false}
          stroke={AXIS_COLOR}
          tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={CATEGORY_WIDTH}
          tick={TICK_STYLE}
          tickLine={false}
          stroke={AXIS_COLOR}
          tickFormatter={truncate}
          interval={0}
        />
        <Tooltip
          cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
          content={<HeadToHeadTooltip />}
        />
        {OUTCOMES.map(({ key, label, color }) => (
          <Bar
            key={key}
            dataKey={key}
            name={label}
            stackId="outcome"
            fill={color}
            stroke="var(--background-color-surface-base)"
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  </Stack>
);
