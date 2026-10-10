// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ChartTooltipRow,
  ChartTooltipSurface,
} from '@nemo/common/src/components/charts/ChartTooltip';
import { AXIS_LABEL_STYLE, GRID_PROPS, TICK_STYLE } from '@nemo/common/src/components/charts/frame';
import { AXIS_COLOR, REFERENCE_LINE_COLOR } from '@nemo/common/src/components/charts/tokens';
import {
  type ComparedEvaluation,
  describeImprovement,
  formatImprovement,
  type ImprovementRow,
  symmetricAxisBound,
} from '@studio/components/charts/EvaluationCompareChart/utils';
import type { FC } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  type TooltipProps,
  XAxis,
  YAxis,
} from 'recharts';

const BAR_SIZE = 12;
const ROW_PADDING = 20;
const AXIS_HEIGHT = 48;
const CATEGORY_WIDTH = 140;

interface ImprovementChartProps {
  baseline: ComparedEvaluation;
  candidates: readonly ComparedEvaluation[];
  rows: readonly ImprovementRow[];
}

const ImprovementTooltip: FC<
  TooltipProps<number, string> & {
    baseline: ComparedEvaluation;
    candidates: readonly ComparedEvaluation[];
  }
> = ({ active, payload, baseline, candidates }) => {
  const row = payload?.[0]?.payload as ImprovementRow | undefined;
  if (!active || !row) return null;
  const format = (value: number | undefined) =>
    value === undefined ? '—' : row.metric.format(value);
  return (
    <ChartTooltipSurface label={row.metric.label}>
      <ChartTooltipRow
        color={baseline.color}
        label={`${baseline.row.name} (baseline)`}
        value={format(row.baseline)}
      />
      {candidates.map(({ row: evaluation, color }) => {
        const { value, improvement } = row.candidates[evaluation.id] ?? {};
        const change = improvement === undefined ? '' : ` (${describeImprovement(improvement)})`;
        return (
          <ChartTooltipRow
            key={evaluation.id}
            color={color}
            label={evaluation.name}
            value={`${format(value)}${change}`}
          />
        );
      })}
    </ChartTooltipSurface>
  );
};

const truncate = (label: string): string => (label.length > 20 ? `${label.slice(0, 19)}…` : label);

export const ImprovementChart: FC<ImprovementChartProps> = ({ baseline, candidates, rows }) => {
  const height = rows.length * (candidates.length * (BAR_SIZE + 2) + ROW_PADDING) + AXIS_HEIGHT;
  const bound = symmetricAxisBound(
    rows.flatMap(({ candidates: values }) =>
      Object.values(values).map(({ improvement }) => improvement ?? 0)
    )
  );
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={[...rows]}
        layout="vertical"
        margin={{ top: 8, right: 24, bottom: 24, left: 0 }}
        barGap={2}
        barSize={BAR_SIZE}
      >
        <CartesianGrid {...GRID_PROPS} horizontal={false} vertical />
        <XAxis
          type="number"
          tick={TICK_STYLE}
          tickLine={false}
          stroke={AXIS_COLOR}
          tickFormatter={formatImprovement}
          domain={[-bound, bound]}
          ticks={[-bound, -bound / 2, 0, bound / 2, bound]}
          label={{
            value: '← Worse · Change vs baseline · Better →',
            position: 'insideBottom',
            offset: -16,
            style: AXIS_LABEL_STYLE,
          }}
        />
        <YAxis
          type="category"
          dataKey={(row: ImprovementRow) => row.metric.label}
          width={CATEGORY_WIDTH}
          tick={TICK_STYLE}
          tickLine={false}
          stroke={AXIS_COLOR}
          tickFormatter={truncate}
          interval={0}
        />
        <ReferenceLine x={0} stroke={REFERENCE_LINE_COLOR} />
        <Tooltip
          cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
          content={<ImprovementTooltip baseline={baseline} candidates={candidates} />}
        />
        {candidates.map(({ row: evaluation, color }) => (
          <Bar
            key={evaluation.id}
            dataKey={(row: ImprovementRow) => row.candidates[evaluation.id]?.improvement}
            name={evaluation.name}
            fill={color}
            radius={4}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
};
