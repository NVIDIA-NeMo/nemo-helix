// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartLegend } from '@nemo/common/src/components/charts/ChartLegend';
import { chartTooltipProps } from '@nemo/common/src/components/charts/ChartTooltip';
import { GRID_PROPS, TICK_STYLE, xAxisLabelProps } from '@nemo/common/src/components/charts/frame';
import { AXIS_TEXT_COLOR } from '@nemo/common/src/components/charts/tokens';
import { Stack, Text } from '@nvidia/foundations-react-core';
import {
  GRADES_BEST_FIRST,
  gradeColor,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type {
  AttributeInferenceMetric,
  MembershipInferenceMetric,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type FC, useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const PIE_HEIGHT = 280;
const ROW_HEIGHT = 36;
const MIN_BAR_CHART_HEIGHT = 200;
const GRADE_LEGEND = GRADES_BEST_FIRST.map((grade) => ({
  id: grade,
  label: grade,
  color: gradeColor(grade),
}));

interface OutcomeSlice {
  grade: string;
  value: number;
  percentage: number | null;
}

const outcomeSlices = (metric: MembershipInferenceMetric) => {
  const useCounts = metric.outcomes.every((outcome) => outcome.count !== null);
  const slices: OutcomeSlice[] = GRADES_BEST_FIRST.flatMap((grade) => {
    const outcome = metric.outcomes.find((candidate) => candidate.grade === grade);
    const value = useCounts ? outcome?.count : outcome?.percentage;
    if (value === null || value === undefined || value === 0) {
      return [];
    }
    return [{ grade, value, percentage: outcome?.percentage ?? null }];
  });
  const total = useCounts ? slices.reduce((sum, slice) => sum + slice.value, 0) : null;
  return { slices, total };
};

interface SliceLabelProps {
  cx: number;
  cy: number;
  midAngle: number;
  outerRadius: number;
  percent: number;
}

const SLICE_LABEL_RADIUS = 0.62;
const MIN_LABELED_SHARE = 0.04;
const RADIANS = Math.PI / 180;

// Tier colors are light in both themes, so slice labels stay dark as in the library report.
const renderSliceLabel = ({ cx, cy, midAngle, outerRadius, percent }: SliceLabelProps) => {
  if (percent < MIN_LABELED_SHARE) {
    return null;
  }
  const radius = outerRadius * SLICE_LABEL_RADIUS;
  return (
    <text
      x={cx + radius * Math.cos(-midAngle * RADIANS)}
      y={cy + radius * Math.sin(-midAngle * RADIANS)}
      fill="#0c0c0c"
      fontSize={12}
      textAnchor="middle"
      dominantBaseline="central"
    >
      {`${(percent * 100).toFixed(1)}%`}
    </text>
  );
};

interface BarLabelProps {
  x?: number | string;
  y?: number | string;
  width?: number | string;
  height?: number | string;
  value?: number | string;
}

const BAR_LABEL_GAP = 6;

/** Drawn as one unwrapped line past the bar end, since recharts wraps labels to the bar's width. */
const renderBarLabel = ({ x, y, width, height, value }: BarLabelProps) => (
  <text
    x={Number(x) + Number(width) + BAR_LABEL_GAP}
    y={Number(y) + Number(height) / 2}
    fill={AXIS_TEXT_COLOR}
    fontSize={11}
    dominantBaseline="central"
  >
    {value}
  </text>
);

/** Membership inference attack outcomes by protection grade. */
export const MembershipInferencePie: FC<{ metric: MembershipInferenceMetric }> = ({ metric }) => {
  const { slices, total } = useMemo(() => outcomeSlices(metric), [metric]);
  const title =
    total === null
      ? 'Membership inference attack outcomes'
      : `Membership inference attack outcomes (${total} attack evaluations graded)`;
  return (
    <Stack gap="density-sm" data-testid="mia-chart">
      <Text kind="label/bold/md">{title}</Text>
      <ResponsiveContainer width="100%" height={PIE_HEIGHT}>
        <PieChart>
          <Pie
            data={slices}
            dataKey="value"
            nameKey="grade"
            outerRadius="85%"
            startAngle={90}
            endAngle={-270}
            stroke="var(--background-color-surface-base)"
            label={renderSliceLabel}
            labelLine={false}
            isAnimationActive={false}
          >
            {slices.map((slice) => (
              <Cell key={slice.grade} fill={gradeColor(slice.grade)} />
            ))}
          </Pie>
          <Tooltip {...chartTooltipProps} />
        </PieChart>
      </ResponsiveContainer>
      <ChartLegend items={GRADE_LEGEND} interactive={false} justify="center" />
    </Stack>
  );
};

const MIN_BAR_WIDTH = 3;

/** Columns least protected first; each carries a label so a zero or missing protection still reads. */
const attributeInferenceRows = (metric: AttributeInferenceMetric) =>
  [...metric.columns]
    .sort(
      (a, b) =>
        (a.protection ?? Number.NEGATIVE_INFINITY) - (b.protection ?? Number.NEGATIVE_INFINITY)
    )
    .map((column) => ({
      ...column,
      plotted: column.protection ?? 0,
      label:
        column.protection === null
          ? 'Unavailable'
          : `${column.protection.toFixed(1)} ${column.grade}`,
    }));

const protectionTooltip = (
  _value: unknown,
  _name: unknown,
  item: { payload?: { label?: string } }
): [string, string] => [item.payload?.label ?? '', 'Protection'];

/** Attribute inference protection per column, least protected first, colored by grade. */
export const AttributeInferenceBars: FC<{ metric: AttributeInferenceMetric }> = ({ metric }) => {
  const rows = useMemo(() => attributeInferenceRows(metric), [metric]);
  const height = Math.max(MIN_BAR_CHART_HEIGHT, rows.length * ROW_HEIGHT + 60);
  return (
    <Stack gap="density-sm" data-testid="aia-chart">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 8, right: 110, bottom: 24, left: 8 }}
        >
          <CartesianGrid {...GRID_PROPS} horizontal={false} vertical />
          <XAxis
            type="number"
            domain={[0, 10]}
            ticks={[0, 2, 4, 6, 8, 10]}
            tick={TICK_STYLE}
            label={xAxisLabelProps('Protection (out of 10)')}
          />
          <YAxis type="category" dataKey="column" tick={TICK_STYLE} width={120} tickLine={false} />
          <Tooltip
            {...chartTooltipProps}
            cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
            formatter={protectionTooltip}
          />
          <Bar
            dataKey="plotted"
            barSize={18}
            minPointSize={MIN_BAR_WIDTH}
            isAnimationActive={false}
          >
            {rows.map((row) => (
              <Cell
                key={row.column}
                fill={row.protection === null ? 'transparent' : gradeColor(row.grade)}
              />
            ))}
            <LabelList dataKey="label" content={renderBarLabel} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ChartLegend items={GRADE_LEGEND} interactive={false} justify="center" />
    </Stack>
  );
};
