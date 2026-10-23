// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HourglassMid } from '@nemo/icons/icons';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Text } from '@nvidia/foundations-react-core';
import { TrendTooltip } from '@studio/components/charts/ExperimentTrendChart/TrendTooltip';
import {
  buildTrendPoints,
  deriveTrendMetrics,
  formatAxisTick,
  formatTimeTick,
} from '@studio/components/charts/ExperimentTrendChart/utils';
import { MetricSelect } from '@studio/components/charts/MetricSelect';
import { useGroupEvaluations } from '@studio/components/charts/useGroupEvaluations';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { type FC, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface ExperimentTrendChartProps {
  workspace: string;
  group: ExperimentResponse;
  /** The group's full evaluation set when the caller already has it loaded (fits one page), so the
   * chart renders from these instead of refetching. */
  preloadedEvaluations?: EvaluationRow[];
  /** True while the caller is still loading rows it will pass via `preloadedEvaluations`; the chart
   * shows a loading state and holds off its own fetch. */
  preloadPending?: boolean;
}

const CHART_HEIGHT = 360;

/**
 * One evaluation metric over time for an experiment group: a point per evaluation, positioned by
 * its `created_at`. The metric is picked from the group's evaluators plus the cost, latency and
 * token rollups, so a group can be read either on quality or on what that quality cost.
 *
 * The series only means anything if the evaluations are comparable — same dataset, same evaluators
 * — which is what the experiment's "Evaluate over time" flag asserts.
 */
export const ExperimentTrendChart: FC<ExperimentTrendChartProps> = ({
  workspace,
  group,
  preloadedEvaluations,
  preloadPending = false,
}) => {
  const hasPreloaded = preloadedEvaluations !== undefined;
  const {
    rows: fetchedRows,
    total: fetchedTotal,
    isLoading: isFetching,
    isError,
  } = useGroupEvaluations(workspace, group.id, { enabled: !hasPreloaded && !preloadPending });
  const points = hasPreloaded ? preloadedEvaluations : fetchedRows;
  const isLoading = hasPreloaded ? false : preloadPending || isFetching;

  // The fetch takes one page, so a group larger than that plots a partial history. A trend that
  // quietly starts partway through reads as the whole story, so say when it does not. The preloaded
  // path is whole by construction — the caller only passes rows when the group fits one page.
  //
  // Counted against the rows fetched, not the points plotted: `buildTrendPoints` also drops rows
  // that carry no value for the selected metric, and folding that into this sentence would report
  // a smaller "most recent" figure than was actually loaded, varying by which metric is selected.
  const omittedCount = hasPreloaded ? 0 : Math.max(0, fetchedTotal - fetchedRows.length);
  const metrics = useMemo(() => deriveTrendMetrics(points), [points]);

  const [metricId, setMetricId] = useState<string | undefined>(undefined);

  // Fall back to the first metric when the selection isn't in the data: the evaluator set is derived
  // from the rows, so a scorer can vanish when the group is refetched. Scorers sort first, so this
  // opens on an evaluator when the group has one and on cost otherwise.
  const metric = metrics.find((m) => m.id === metricId) ?? metrics[0];

  const plotPoints = useMemo(
    () => (metric ? buildTrendPoints(points, metric) : []),
    [points, metric]
  );

  const renderBody = () => {
    if (isError) {
      return <Text kind="body/regular/sm">Could not load the evaluations for this group.</Text>;
    }
    if (isLoading) {
      return (
        <div aria-busy className="flex h-[360px] items-center justify-center gap-2">
          <HourglassMid width={20} height={20} className="animate-spin text-brand" />
          <Text kind="body/regular/sm" color="subtle">
            Loading evaluations…
          </Text>
        </div>
      );
    }
    if (plotPoints.length === 0 || !metric) {
      return (
        <Text kind="body/regular/sm" color="subtle">
          No evaluations with the selected metric to plot over time.
        </Text>
      );
    }
    return (
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart margin={{ top: 16, right: 24, bottom: 24, left: 16 }} data={plotPoints}>
          <CartesianGrid stroke="var(--border-color-base)" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="Run date"
            // A time axis has to be scaled by value, not by index, or evaluations an hour apart are
            // spread as widely as ones a month apart.
            scale="time"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'Run date', position: 'insideBottom', offset: -12 }}
            tickFormatter={formatTimeTick}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={metric.label}
            label={{ value: metric.label, angle: -90, position: 'insideLeft' }}
            tickFormatter={formatAxisTick}
            tick={{ fontSize: 11 }}
          />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<TrendTooltip metric={metric} />} />
          <Line
            type="linear"
            dataKey="y"
            name={metric.label}
            stroke="var(--border-color-brand)"
            strokeWidth={2}
            // Dots carry the tooltip, and each one is a distinct evaluation worth pointing at.
            dot={{ fill: 'var(--border-color-brand)', r: 3 }}
            activeDot={{ r: 5 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="flex flex-col gap-3 rounded border border-base bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <Text kind="title/xs">{`${metric?.label ?? 'Metric'} over time`}</Text>
          {omittedCount > 0 && (
            <Text kind="body/regular/xs" color="subtle">
              {`Showing the ${fetchedRows.length.toLocaleString()} most recent of ${fetchedTotal.toLocaleString()} evaluations.`}
            </Text>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <MetricSelect
            label="Metric"
            value={metric?.id ?? ''}
            metrics={metrics}
            onChange={setMetricId}
            triggerClassName="w-44"
          />
        </div>
      </div>
      {renderBody()}
    </div>
  );
};
