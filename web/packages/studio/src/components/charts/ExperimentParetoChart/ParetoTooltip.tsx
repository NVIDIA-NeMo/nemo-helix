// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type {
  ParetoMetric,
  ParetoPlotPoint,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import type { FC } from 'react';

/** Format a metric value for tooltips: cost as USD, latency in ms, whole tokens, evaluator scores as-is. */
const formatMetricValue = (metric: ParetoMetric, value: number): string => {
  if (metric.id === 'cost_usd') {
    return `$${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}`;
  }
  if (metric.id === 'latency_ms') {
    return `${Math.round(value).toLocaleString()} ms`;
  }
  if (metric.id === 'tokens') {
    return Math.round(value).toLocaleString();
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
};

interface ParetoTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{ payload: ParetoPlotPoint }>;
  xMetric: ParetoMetric;
  yMetric: ParetoMetric;
}

export const ParetoTooltip: FC<ParetoTooltipProps> = ({ active, payload, xMetric, yMetric }) => {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="rounded border border-base bg-surface p-2 shadow-md">
      <Text kind="body/semibold/sm">{point.name}</Text>
      <div className="mt-1 flex flex-col gap-0.5">
        <Text kind="body/regular/xs">
          {xMetric.label}: {formatMetricValue(xMetric, point.x)}
        </Text>
        <Text kind="body/regular/xs">
          {yMetric.label}: {formatMetricValue(yMetric, point.y)}
        </Text>
        {point.onFrontier && (
          <Text kind="body/regular/xs" color="brand">
            On the Pareto frontier
          </Text>
        )}
      </div>
    </div>
  );
};
