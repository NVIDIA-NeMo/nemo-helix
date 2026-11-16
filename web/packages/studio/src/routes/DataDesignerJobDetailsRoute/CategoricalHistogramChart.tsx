// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { chartTooltipProps } from '@nemo/common/src/components/charts/ChartTooltip';
import { Stack, Text } from '@nvidia/foundations-react-core';
import type { CategoricalHistogramData } from '@studio/routes/DataDesignerJobDetailsRoute/datasetProfilerTypes';
import { type FC, useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface CategoricalHistogramChartProps {
  histogram: CategoricalHistogramData;
}

/** Show at most this many bars; remaining categories are summarized below. */
const MAX_BARS = 12;
const CHART_HEIGHT = 220;
const TICK_STYLE = { fontSize: 11, fill: 'var(--text-color-base)' } as const;

interface HistogramBar {
  label: string;
  count: number;
}

/** Truncate long category labels so the axis stays legible. */
const truncateLabel = (label: string): string =>
  label.length > 14 ? `${label.slice(0, 13)}…` : label;

/**
 * Vertical bar chart of a categorical sampler column's value distribution.
 * Bars are sorted by frequency and capped at {@link MAX_BARS}; any overflow is
 * surfaced as a "+N more categories" note so the chart stays readable.
 */
export const CategoricalHistogramChart: FC<CategoricalHistogramChartProps> = ({ histogram }) => {
  const { bars, hiddenCount, hiddenTotal } = useMemo(() => {
    const all: HistogramBar[] = histogram.categories.map((category, index) => ({
      label: String(category),
      count: histogram.counts[index] ?? 0,
    }));
    all.sort((a, b) => b.count - a.count);
    const visible = all.slice(0, MAX_BARS);
    const hidden = all.slice(MAX_BARS);
    return {
      bars: visible,
      hiddenCount: hidden.length,
      hiddenTotal: hidden.reduce((sum, bar) => sum + bar.count, 0),
    };
  }, [histogram]);

  if (bars.length === 0) {
    return (
      <Text kind="body/regular/sm" className="text-muted">
        No category counts available.
      </Text>
    );
  }

  return (
    <Stack gap="density-sm">
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <BarChart data={bars} margin={{ top: 16, bottom: 8, left: 0, right: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-color-base)" />
          <XAxis
            dataKey="label"
            tick={TICK_STYLE}
            tickFormatter={truncateLabel}
            tickLine={false}
            interval={0}
            angle={-35}
            textAnchor="end"
            height={56}
          />
          <YAxis tick={TICK_STYLE} width={40} allowDecimals={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
            {...chartTooltipProps}
            formatter={(value: number) => [value.toLocaleString(), 'Count']}
          />
          <Bar dataKey="count" name="Count" fill="var(--text-color-brand)" radius={[4, 4, 0, 0]}>
            <LabelList
              dataKey="count"
              position="top"
              fill="var(--text-color-base)"
              fontSize={11}
              formatter={(value: number) => value.toLocaleString()}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {hiddenCount > 0 && (
        <Text kind="body/regular/sm" className="text-muted">
          +{hiddenCount} more {hiddenCount === 1 ? 'category' : 'categories'} (
          {hiddenTotal.toLocaleString()} records)
        </Text>
      )}
    </Stack>
  );
};
