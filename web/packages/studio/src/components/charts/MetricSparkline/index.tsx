// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { MetricTrendPoint } from '@studio/components/charts/MetricTrend';
import type { FC } from 'react';
import { Line, LineChart, YAxis } from 'recharts';

export interface MetricSparklineProps {
  points: MetricTrendPoint[];
  width?: number;
  height?: number;
}

/** Axis-free trendline sized for a table cell. Direction only; the detail page has the values. */
export const MetricSparkline: FC<MetricSparklineProps> = ({ points, width = 120, height = 28 }) => {
  if (points.length < 2) return <Text className="text-secondary">-</Text>;
  const falling = points[points.length - 1].value < points[0].value;
  return (
    <LineChart width={width} height={height} data={points} aria-hidden>
      <YAxis domain={['dataMin', 'dataMax']} hide />
      <Line
        type="linear"
        dataKey="value"
        stroke={falling ? 'var(--text-color-accent-red)' : 'var(--text-color-brand)'}
        strokeWidth={1.5}
        dot={false}
        isAnimationActive={false}
      />
    </LineChart>
  );
};
