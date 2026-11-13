// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChartTooltipSurface } from '@nemo/common/src/components/charts/ChartTooltip';
import { Text } from '@nvidia/foundations-react-core';
import {
  formatTimestamp,
  type TrendMetric,
  type TrendPlotPoint,
} from '@studio/components/charts/ExperimentTrendChart/utils';
import type { FC } from 'react';

interface TrendTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: TrendPlotPoint }>;
  metric: TrendMetric;
}

export const TrendTooltip: FC<TrendTooltipProps> = ({ active, payload, metric }) => {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <ChartTooltipSurface label={point.name}>
      <div className="flex flex-col gap-0.5">
        <Text kind="body/regular/xs">
          {metric.label}: {metric.format(point.y)}
        </Text>
        <Text kind="body/regular/xs" color="subtle">
          {formatTimestamp(point.x)}
        </Text>
      </div>
    </ChartTooltipSurface>
  );
};
