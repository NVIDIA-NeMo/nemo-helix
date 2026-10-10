// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ChartTooltipRow,
  ChartTooltipSurface,
} from '@nemo/common/src/components/charts/ChartTooltip';
import { GRID_PROPS, TICK_STYLE } from '@nemo/common/src/components/charts/frame';
import { AXIS_COLOR } from '@nemo/common/src/components/charts/tokens';
import type {
  ComparedEvaluation,
  MetricValuesRow,
} from '@studio/components/charts/EvaluationCompareChart/utils';
import { formatAxisTick } from '@studio/components/charts/ExperimentTrendChart/utils';
import type { FC } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  type TooltipProps,
  XAxis,
  YAxis,
} from 'recharts';

const CHART_HEIGHT = 280;

interface ScoresChartProps {
  evaluations: readonly ComparedEvaluation[];
  rows: readonly MetricValuesRow[];
}

const ScoresTooltip: FC<
  TooltipProps<number, string> & { evaluations: readonly ComparedEvaluation[] }
> = ({ active, payload, evaluations }) => {
  const row = payload?.[0]?.payload as MetricValuesRow | undefined;
  if (!active || !row) return null;
  return (
    <ChartTooltipSurface label={row.metric.label}>
      {evaluations.map(({ row: evaluation, color }) => {
        const value = row.values[evaluation.id];
        return (
          <ChartTooltipRow
            key={evaluation.id}
            color={color}
            label={evaluation.name}
            value={value === undefined ? '—' : row.metric.format(value)}
          />
        );
      })}
    </ChartTooltipSurface>
  );
};

export const ScoresChart: FC<ScoresChartProps> = ({ evaluations, rows }) => (
  <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
    <BarChart data={[...rows]} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barGap={2}>
      <CartesianGrid {...GRID_PROPS} />
      <XAxis
        dataKey={(row: MetricValuesRow) => row.metric.label}
        tick={TICK_STYLE}
        tickLine={false}
        stroke={AXIS_COLOR}
        interval={0}
      />
      <YAxis
        tick={TICK_STYLE}
        tickLine={false}
        stroke={AXIS_COLOR}
        tickFormatter={formatAxisTick}
      />
      <Tooltip
        cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
        content={<ScoresTooltip evaluations={evaluations} />}
      />
      {evaluations.map(({ row: evaluation, color }) => (
        <Bar
          key={evaluation.id}
          dataKey={(row: MetricValuesRow) => row.values[evaluation.id]}
          name={evaluation.name}
          fill={color}
          radius={[4, 4, 0, 0]}
          maxBarSize={32}
          isAnimationActive={false}
        />
      ))}
    </BarChart>
  </ResponsiveContainer>
);
