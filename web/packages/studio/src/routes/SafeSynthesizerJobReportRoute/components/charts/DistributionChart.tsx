// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { chartTooltipProps } from '@nemo/common/src/components/charts/ChartTooltip';
import { GRID_PROPS, TICK_STYLE } from '@nemo/common/src/components/charts/frame';
import { Stack, Text } from '@nvidia/foundations-react-core';
import {
  binnedRows,
  categoryRows,
  DATASET_COLORS,
  DATASET_LABELS,
  formatNumber,
  formatPercent,
  hasSyntheticValues,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type { Distribution } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type FC, useMemo } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const DEFAULT_HEIGHT = 220;
const BAR_OPACITY = 0.7;
const AREA_OPACITY = 0.35;

interface DistributionChartProps {
  title?: string;
  distribution: Distribution | null;
  height?: number;
}

const percentTooltip = (value: unknown) => formatPercent(typeof value === 'number' ? value : null);

const CategoryBars: FC<{ distribution: Distribution; height: number }> = ({
  distribution,
  height,
}) => {
  const rows = useMemo(() => categoryRows(distribution), [distribution]);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} barGap={1} barCategoryGap="20%">
        <CartesianGrid {...GRID_PROPS} />
        <XAxis dataKey="label" tick={TICK_STYLE} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={TICK_STYLE} tickLine={false} width={36} unit="%" />
        <Tooltip
          {...chartTooltipProps}
          cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
          formatter={percentTooltip}
        />
        <Bar
          dataKey="training"
          name={DATASET_LABELS.training}
          fill={DATASET_COLORS.training}
          fillOpacity={BAR_OPACITY}
          isAnimationActive={false}
        />
        <Bar
          dataKey="synthetic"
          name={DATASET_LABELS.synthetic}
          fill={DATASET_COLORS.synthetic}
          fillOpacity={BAR_OPACITY}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
};

const BinnedAreas: FC<{ distribution: Distribution; height: number }> = ({
  distribution,
  height,
}) => {
  const rows = useMemo(() => binnedRows(distribution), [distribution]);
  const edges = distribution.bin_edges ?? [];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows}>
        <CartesianGrid {...GRID_PROPS} />
        <XAxis
          dataKey="edge"
          type="number"
          domain={[edges[0], edges[edges.length - 1]]}
          tick={TICK_STYLE}
          tickFormatter={(value: number) => formatNumber(value, 4)}
          tickLine={false}
          allowDataOverflow
        />
        <YAxis tick={TICK_STYLE} tickLine={false} width={36} unit="%" />
        <Tooltip
          {...chartTooltipProps}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.label ?? ''}
          formatter={percentTooltip}
        />
        <Area
          type="stepAfter"
          dataKey="training"
          name={DATASET_LABELS.training}
          stroke={DATASET_COLORS.training}
          fill={DATASET_COLORS.training}
          fillOpacity={AREA_OPACITY}
          isAnimationActive={false}
        />
        <Area
          type="stepAfter"
          dataKey="synthetic"
          name={DATASET_LABELS.synthetic}
          stroke={DATASET_COLORS.synthetic}
          fill={DATASET_COLORS.synthetic}
          fillOpacity={AREA_OPACITY}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

/** Training and synthetic percentages per category, or per bin overlaid like a histogram. */
export const DistributionChart: FC<DistributionChartProps> = ({
  title,
  distribution,
  height = DEFAULT_HEIGHT,
}) => {
  const isBinned = distribution?.bin_edges !== null && distribution?.bin_edges !== undefined;
  return (
    <Stack gap="density-xs" data-testid="distribution-chart">
      {title ? <Text kind="label/bold/md">{title}</Text> : null}
      {distribution === null ? (
        <Text kind="body/regular/sm" className="text-secondary">
          No numeric training values to chart
        </Text>
      ) : null}
      {distribution !== null && isBinned ? (
        <BinnedAreas distribution={distribution} height={height} />
      ) : null}
      {distribution !== null && !isBinned ? (
        <CategoryBars distribution={distribution} height={height} />
      ) : null}
      {distribution !== null && isBinned && !hasSyntheticValues(distribution) ? (
        <Text kind="body/regular/sm" className="text-secondary">
          No numeric synthetic values
        </Text>
      ) : null}
    </Stack>
  );
};
