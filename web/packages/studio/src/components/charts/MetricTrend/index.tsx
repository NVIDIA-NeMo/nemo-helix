// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type DeltaTone,
  deltaTone,
  formatSignedDelta,
} from '@nemo/common/src/components/DeltaText';
import { Flex, Stack, Tag, Text } from '@nvidia/foundations-react-core';
import { SeriesButtonGroup } from '@studio/components/charts/MetricTrend/SeriesButtonGroup';
import { useNvColorMode } from '@studio/components/DagCanvas/useNvColorMode';
import { StackedSkeleton } from '@studio/components/StackedSkeleton';
import { Triangle } from 'lucide-react';
import { FC, useId, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface MetricTrendPoint {
  /** X-axis label for the point, shown in the tooltip. */
  label: string;
  value: number;
}

export interface MetricTrendSeries {
  id: string;
  /** Pill label, e.g. "Solved". */
  label: string;
  value: number;
  /** Change over the compared period. Positive renders up, negative down. */
  delta?: number;
  points: MetricTrendPoint[];
}

export interface MetricTrendProps {
  series: MetricTrendSeries[];
  /** Names the pill group for screen readers — whatever the trend is measuring. */
  label: string;
  /** Qualifies the delta, e.g. "vs. 7 days ago". */
  comparisonLabel?: string;
  /** Caption under the value, e.g. "Latest result". */
  valueLabel?: string;
  /** Controls the active pill. Leave undefined to let the trend manage it. */
  selectedSeriesId?: string;
  onSeriesChange?: (seriesId: string) => void;
  formatValue?: (value: number) => string;
  formatDelta?: (delta: number) => string;
  chartHeight?: number;
  isPending?: boolean;
  /** Applied to the outer row, for callers that need the chart to bleed into their padding. */
  className?: string;
}

const DEFAULT_CHART_HEIGHT = 160;

/**
 * The area fill sits on a near-black surface in dark mode, so the light-mode opacities
 * wash out to almost nothing. Carry more of the accent color, and keep a floor at the
 * bottom of the gradient so the fill stays readable all the way down.
 */
const AREA_GRADIENT = {
  dark: { top: 0.65, mid: 0.3, bottom: 0.08 },
  light: { top: 0.3, mid: 0.12, bottom: 0 },
} as const;

const formatPercent = (value: number): string => `${value.toFixed(1)}%`;

const formatTrendDelta = (delta: number): string => formatSignedDelta(delta, 1);

/** The chip form of the same green/red/muted scale {@link DeltaText} paints as bare text. */
const TAG_COLOR: Record<DeltaTone, 'green' | 'red' | 'gray'> = {
  improved: 'green',
  regressed: 'red',
  unchanged: 'gray',
};

/**
 * The latest value for the selected series, its change over the compared period, and a trendline
 * of every point behind it. Standalone so it can sit inside any container; {@link MetricTrendPanel}
 * wraps it in a titled panel.
 */
export const MetricTrend: FC<MetricTrendProps> = ({
  series,
  label,
  comparisonLabel,
  valueLabel,
  selectedSeriesId,
  onSeriesChange,
  formatValue = formatPercent,
  formatDelta = formatTrendDelta,
  chartHeight = DEFAULT_CHART_HEIGHT,
  isPending = false,
  className,
}) => {
  // useId() emits colons, which are not valid inside an SVG `url(#...)` reference.
  const gradientId = `metric-trend-${useId().replace(/:/g, '')}`;
  const [internalSeriesId, setInternalSeriesId] = useState<string | undefined>(series[0]?.id);

  const activeId = selectedSeriesId ?? internalSeriesId;
  const active = useMemo(
    () => series.find((s) => s.id === activeId) ?? series[0],
    [series, activeId]
  );

  const selectSeries = (seriesId: string) => {
    if (selectedSeriesId === undefined) {
      setInternalSeriesId(seriesId);
    }
    onSeriesChange?.(seriesId);
  };

  const delta = active?.delta;
  const isNegative = delta !== undefined && delta < 0;
  const isZero = delta === 0;
  const deltaColor = TAG_COLOR[deltaTone(delta ?? 0)];
  const lineColor = isNegative ? 'var(--text-color-accent-red)' : 'var(--text-color-brand)';
  const colorMode = useNvColorMode();
  const gradient = colorMode === 'dark' ? AREA_GRADIENT.dark : AREA_GRADIENT.light;
  // A single datapoint has no line to draw, so an AreaChart shows only a lone dot. Render a
  // flat ReferenceLine across the surface instead, per the design.
  const isSingle = active?.points.length === 1;

  return (
    <Flex align="center" gap="density-2xl" className={className}>
      <Stack gap="density-xs" className="shrink-0 pb-density-2xl text-center">
        <Text kind="display/lg">{active ? formatValue(active.value) : '—'}</Text>

        {delta !== undefined && (
          <Flex align="center" gap="density-sm">
            <Tag readOnly color={deltaColor} density="compact">
              {!isZero && (
                <Triangle
                  size={12}
                  className={`fill-current ${isNegative ? 'rotate-180' : ''}`}
                  aria-hidden
                />
              )}
              {formatDelta(delta)}
            </Tag>
            {comparisonLabel && <Text kind="body/regular/md">{comparisonLabel}</Text>}
          </Flex>
        )}

        {valueLabel && <Text kind="body/regular/md">{valueLabel}</Text>}
      </Stack>

      <Stack gap="density-sm" className="min-w-0 flex-1">
        {series.length > 1 && (
          <SeriesButtonGroup
            series={series}
            activeId={active?.id}
            label={label}
            onSelect={selectSeries}
          />
        )}

        <div className="overflow-hidden rounded-br-density-xl">
          {isPending ? (
            <StackedSkeleton count={1} height={chartHeight} className="w-full" />
          ) : !active ? (
            // eslint-disable-next-line no-restricted-syntax -- chartHeight is a caller-controlled number, not a static Tailwind class
            <Flex align="center" justify="center" style={{ height: chartHeight }}>
              <Text kind="body/regular/md" className="text-secondary">
                No data available
              </Text>
            </Flex>
          ) : (
            <ResponsiveContainer width="100%" height={chartHeight}>
              <AreaChart data={active.points} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={lineColor} stopOpacity={gradient.top} />
                    <stop offset="55%" stopColor={lineColor} stopOpacity={gradient.mid} />
                    <stop offset="100%" stopColor={lineColor} stopOpacity={gradient.bottom} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="label" hide />
                <YAxis
                  domain={
                    isSingle
                      ? [active.points[0].value - 1, active.points[0].value + 1]
                      : ['dataMin', 'dataMax']
                  }
                  hide
                />
                <Tooltip
                  cursor={{ stroke: 'var(--border-color-base)', strokeWidth: 1 }}
                  formatter={(value: number) => [formatValue(value), active.label]}
                  contentStyle={{
                    fontSize: 12,
                    backgroundColor: 'var(--background-color-component-tooltip)',
                    borderColor: 'var(--border-color-base)',
                    color: 'var(--text-color-base)',
                  }}
                  labelStyle={{ color: 'var(--text-color-base)' }}
                  itemStyle={{ color: 'var(--text-color-base)' }}
                />
                <Area
                  type="linear"
                  dataKey="value"
                  name={active.label}
                  stroke={lineColor}
                  strokeWidth={2}
                  fill={`url(#${gradientId})`}
                  dot={!isSingle && active.points.length <= 2}
                  isAnimationActive={false}
                />
                {isSingle && (
                  <ReferenceLine
                    y={active.points[0].value}
                    stroke={lineColor}
                    strokeWidth={2}
                    ifOverflow="extendDomain"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </Stack>
    </Flex>
  );
};
