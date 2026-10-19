// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getGetExperimentQueryKey,
  useUpdateExperiment,
} from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Button, Text } from '@nvidia/foundations-react-core';
import { ParetoTooltip } from '@studio/components/charts/ExperimentParetoChart/ParetoTooltip';
import {
  buildParetoPoints,
  deriveParetoMetrics,
  resolveParetoAxes,
} from '@studio/components/charts/ExperimentParetoChart/utils';
import { MetricSelect } from '@studio/components/charts/MetricSelect';
import { useGroupEvaluations } from '@studio/components/charts/useGroupEvaluations';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { useQueryClient } from '@tanstack/react-query';
import { Loader2, Save } from 'lucide-react';
import { type FC, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface ExperimentParetoChartProps {
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
const DEFAULT_X_METRIC = 'cost_usd';
const DEFAULT_Y_METRIC = 'latency_ms';

/** Compact big tick values so they don't collide with the axis title (16000 -> "16K"); keep small
 * values precise so close cost/score ticks stay distinct. */
const formatAxisTick = (value: number): string =>
  Math.abs(value) >= 1000
    ? value.toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: 1 })
    : value.toLocaleString(undefined, { maximumFractionDigits: 3 });

/**
 * Pareto view for an experiment group: one point per evaluation, frontier highlighted. The two axes
 * are picked from the group's metrics and persisted on the group (shared across viewers).
 */
export const ExperimentParetoChart: FC<ExperimentParetoChartProps> = ({
  workspace,
  group,
  preloadedEvaluations,
  preloadPending = false,
}) => {
  const queryClient = useQueryClient();
  const toast = useToast();

  const hasPreloaded = preloadedEvaluations !== undefined;
  const {
    rows: fetchedRows,
    isLoading: isFetching,
    isError,
  } = useGroupEvaluations(workspace, group.id, { enabled: !hasPreloaded && !preloadPending });
  const points = hasPreloaded ? preloadedEvaluations : fetchedRows;
  const isLoading = hasPreloaded ? false : preloadPending || isFetching;
  const metrics = useMemo(() => deriveParetoMetrics(points), [points]);

  const savedX = group.pareto?.x_metric ?? DEFAULT_X_METRIC;
  const savedY = group.pareto?.y_metric ?? DEFAULT_Y_METRIC;
  const [pickedXId, setPickedXId] = useState<string>();
  const [pickedYId, setPickedYId] = useState<string>();
  const defaultAxes = useMemo(
    () => resolveParetoAxes(points, metrics, savedX, savedY),
    [points, metrics, savedX, savedY]
  );
  const xMetric = metrics.find((m) => m.id === pickedXId) ?? defaultAxes?.x;
  const yMetric = metrics.find((m) => m.id === pickedYId) ?? defaultAxes?.y;

  const { mutate: saveGroup, isPending: isSaving } = useUpdateExperiment({
    mutation: {
      onSuccess: () => {
        toast.success('Saved the group default Pareto view.');
        queryClient.invalidateQueries({
          queryKey: getGetExperimentQueryKey(workspace, group.name),
        });
      },
      onError: () => toast.error('Failed to save the Pareto metrics.'),
    },
  });

  // PUT is a full replace, so send every field — only `pareto` changes.
  const persistAxes = (xMetric: string, yMetric: string) => {
    saveGroup({
      workspace,
      name: group.name,
      data: {
        name: group.name,
        description: group.description,
        insight_id: group.insight_id,
        summary: group.summary,
        metadata: group.metadata,
        default_sort: group.default_sort,
        pareto: { x_metric: xMetric, y_metric: yMetric },
      },
    });
  };

  const hasPickedAxes = pickedXId !== undefined || pickedYId !== undefined;
  const hasUnsavedAxes =
    hasPickedAxes &&
    xMetric !== undefined &&
    yMetric !== undefined &&
    (xMetric.id !== savedX || yMetric.id !== savedY);

  const plotPoints = useMemo(
    () => (xMetric && yMetric ? buildParetoPoints(points, xMetric, yMetric) : []),
    [points, xMetric, yMetric]
  );

  // Frontier points are sorted by x so Recharts draws the connecting line along the frontier curve.
  const frontierPoints = useMemo(
    () => plotPoints.filter((p) => p.onFrontier).sort((a, b) => a.x - b.x),
    [plotPoints]
  );
  const dominatedPoints = useMemo(() => plotPoints.filter((p) => !p.onFrontier), [plotPoints]);

  const renderBody = () => {
    if (isError) {
      return <Text kind="body/regular/sm">Could not load the Pareto data for this group.</Text>;
    }
    if (isLoading) {
      return (
        <div aria-busy className="flex h-[360px] items-center justify-center gap-2">
          <Loader2 width={20} height={20} className="animate-spin text-brand" />
          <Text kind="body/regular/sm" color="subtle">
            Loading evaluations…
          </Text>
        </div>
      );
    }
    if (plotPoints.length === 0 || !xMetric || !yMetric) {
      return (
        <Text kind="body/regular/sm" color="subtle">
          No evaluations with both selected metrics to plot.
        </Text>
      );
    }
    return (
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 16 }}>
          <CartesianGrid stroke="var(--border-color-base)" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name={xMetric.label}
            label={{ value: xMetric.label, position: 'insideBottom', offset: -12 }}
            tickFormatter={formatAxisTick}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yMetric.label}
            label={{ value: yMetric.label, angle: -90, position: 'insideLeft' }}
            tickFormatter={formatAxisTick}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={<ParetoTooltip xMetric={xMetric} yMetric={yMetric} />}
          />
          <Scatter
            name="Evaluations"
            data={dominatedPoints}
            fill="var(--text-color-secondary)"
            fillOpacity={0.7}
          />
          <Scatter
            name="Pareto frontier"
            data={frontierPoints}
            fill="var(--border-color-brand)"
            line={{ stroke: 'var(--border-color-brand)' }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    );
  };

  return (
    <div className="flex flex-col gap-3 rounded border border-base bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <Text kind="title/xs">
          {xMetric && yMetric ? `${xMetric.label} vs. ${yMetric.label}` : null}
        </Text>
        <div className="flex flex-wrap items-center gap-4">
          <MetricSelect
            label="X axis"
            value={xMetric?.id ?? ''}
            metrics={metrics}
            onChange={setPickedXId}
          />
          <MetricSelect
            label="Y axis"
            value={yMetric?.id ?? ''}
            metrics={metrics}
            onChange={setPickedYId}
          />
          <Button
            kind="primary"
            size="small"
            aria-label={isSaving ? 'Saving' : 'Save as group default'}
            disabled={!hasUnsavedAxes || isSaving}
            onClick={() => xMetric && yMetric && persistAxes(xMetric.id, yMetric.id)}
          >
            {isSaving ? (
              <Loader2 width={16} height={16} className="animate-spin" />
            ) : (
              <Save width={16} height={16} />
            )}
          </Button>
        </div>
      </div>
      {renderBody()}
    </div>
  );
};
