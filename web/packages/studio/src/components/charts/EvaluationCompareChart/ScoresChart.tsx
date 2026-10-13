// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ChartTooltipRow,
  ChartTooltipSurface,
} from '@nemo/common/src/components/charts/ChartTooltip';
import { GRID_PROPS, TICK_STYLE } from '@nemo/common/src/components/charts/frame';
import { AXIS_COLOR } from '@nemo/common/src/components/charts/tokens';
import { Grid, Stack, Text } from '@nvidia/foundations-react-core';
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

const PANEL_HEIGHT = 220;

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

interface ScorePanelProps {
  evaluations: readonly ComparedEvaluation[];
  row: MetricValuesRow;
}

const ScorePanel: FC<ScorePanelProps> = ({ evaluations, row }) => (
  <Stack gap="density-xs">
    <Text kind="label/semibold/sm">{row.metric.label}</Text>
    <ResponsiveContainer width="100%" height={PANEL_HEIGHT}>
      <BarChart data={[row]} margin={{ top: 8, right: 16, bottom: 0, left: 0 }} barGap={4}>
        <CartesianGrid {...GRID_PROPS} />
        <XAxis dataKey={() => row.metric.label} hide />
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
            dataKey={(metricRow: MetricValuesRow) => metricRow.values[evaluation.id]}
            name={evaluation.name}
            fill={color}
            radius={[4, 4, 0, 0]}
            maxBarSize={48}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  </Stack>
);

/** One panel per evaluator, each on its own scale, so a 0–1 score isn't flattened next to a 0–100 one. */
export const ScoresChart: FC<ScoresChartProps> = ({ evaluations, rows }) => (
  <Grid cols={{ base: 1, md: 2, xl: 3 }} gap="density-xl">
    {rows.map((row) => (
      <ScorePanel key={row.metric.id} evaluations={evaluations} row={row} />
    ))}
  </Grid>
);
