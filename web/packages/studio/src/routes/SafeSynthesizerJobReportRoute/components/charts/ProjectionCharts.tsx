// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { chartTooltipProps } from '@nemo/common/src/components/charts/ChartTooltip';
import {
  GRID_PROPS,
  TICK_STYLE,
  xAxisLabelProps,
  yAxisLabelProps,
} from '@nemo/common/src/components/charts/frame';
import { Grid, Stack, Text } from '@nvidia/foundations-react-core';
import {
  autoBinEdges,
  DATASET_COLORS,
  DATASET_LABELS,
  finiteValues,
  overlaidHistogramRows,
  pointsDomain,
  type ScatterPoint,
  toPoints,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type { Projection } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type FC, useMemo } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const CHART_HEIGHT = 280;
const MARGIN = { top: 8, right: 16, bottom: 24, left: 8 };
const DOT_OPACITY = 0.6;
const DOT_RADIUS = 2;
const AREA_OPACITY = 0.35;

type Dataset = 'training' | 'synthetic';

interface ScatterSeries {
  dataset: Dataset;
  points: ScatterPoint[];
}

const formatAxisValue = (value: number) => value.toFixed(2);

const Dot: FC<{ cx?: number; cy?: number; fill?: string }> = ({ cx, cy, fill }) => (
  <circle cx={cx} cy={cy} r={DOT_RADIUS} fill={fill} fillOpacity={DOT_OPACITY} />
);

interface ScatterPanelProps {
  title?: string;
  series: ScatterSeries[];
  xLabel: string;
  yLabel: string;
  xDomain: [number, number];
  yDomain: [number, number];
}

const ScatterPanel: FC<ScatterPanelProps> = ({
  title,
  series,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
}) => (
  <Stack gap="density-xs">
    {title ? <Text kind="label/bold/md">{title}</Text> : null}
    <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
      <ScatterChart margin={MARGIN}>
        <CartesianGrid {...GRID_PROPS} />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          domain={xDomain}
          tick={TICK_STYLE}
          tickFormatter={formatAxisValue}
          label={xAxisLabelProps(xLabel)}
          allowDataOverflow
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          domain={yDomain}
          tick={TICK_STYLE}
          tickFormatter={formatAxisValue}
          label={yAxisLabelProps(yLabel)}
          width={48}
          allowDataOverflow
        />
        {series.map(({ dataset, points }) => (
          <Scatter
            key={dataset}
            name={DATASET_LABELS[dataset]}
            data={points}
            fill={DATASET_COLORS[dataset]}
            shape={<Dot />}
            isAnimationActive={false}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  </Stack>
);

const projectionPoints = (projection: Projection) => {
  const training = toPoints(projection.training.x, projection.training.y);
  const synthetic = toPoints(projection.synthetic.x, projection.synthetic.y);
  const all = [...training, ...synthetic];
  return {
    training,
    synthetic,
    xDomain: pointsDomain(all, 'x'),
    yDomain: pointsDomain(all, 'y'),
  };
};

/** Training and synthetic records on the same two principal components, side by side. */
export const DeepStructureScatter: FC<{ projection: Projection }> = ({ projection }) => {
  const { training, synthetic, xDomain, yDomain } = useMemo(
    () => projectionPoints(projection),
    [projection]
  );
  const [xLabel, yLabel] = projection.axes;
  return (
    <Grid cols={{ xs: 1, md: 2 }} gap="density-lg" data-testid="deep-structure-scatter">
      <ScatterPanel
        title={DATASET_LABELS.training}
        series={[{ dataset: 'training', points: training }]}
        xLabel={xLabel}
        yLabel={yLabel}
        xDomain={xDomain}
        yDomain={yDomain}
      />
      <ScatterPanel
        title={DATASET_LABELS.synthetic}
        series={[{ dataset: 'synthetic', points: synthetic }]}
        xLabel={xLabel}
        yLabel={yLabel}
        xDomain={xDomain}
        yDomain={yDomain}
      />
    </Grid>
  );
};

const ComponentHistogram: FC<{ label: string; training: number[]; synthetic: number[] }> = ({
  label,
  training,
  synthetic,
}) => {
  const rows = useMemo(
    () => overlaidHistogramRows(training, synthetic, autoBinEdges(training, synthetic), 'density'),
    [training, synthetic]
  );
  return (
    <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
      <AreaChart data={rows} margin={MARGIN}>
        <CartesianGrid {...GRID_PROPS} />
        <XAxis
          dataKey="edge"
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={TICK_STYLE}
          tickFormatter={formatAxisValue}
          label={xAxisLabelProps(label)}
        />
        <YAxis tick={TICK_STYLE} tickFormatter={formatAxisValue} width={48} />
        <Tooltip
          {...chartTooltipProps}
          labelFormatter={(_, payload) => payload?.[0]?.payload?.label ?? ''}
          formatter={(value: number) => value.toFixed(3)}
        />
        {(['training', 'synthetic'] as const).map((dataset) => (
          <Area
            key={dataset}
            type="stepAfter"
            dataKey={dataset}
            name={DATASET_LABELS[dataset]}
            stroke={DATASET_COLORS[dataset]}
            fill={DATASET_COLORS[dataset]}
            fillOpacity={AREA_OPACITY}
            isAnimationActive={false}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

/** A text column's embeddings: density of each component and the two components against each other. */
export const TextSemanticCharts: FC<{ column: string; projection: Projection }> = ({
  column,
  projection,
}) => {
  const { training, synthetic, xDomain, yDomain } = useMemo(
    () => projectionPoints(projection),
    [projection]
  );
  const [xLabel, yLabel] = projection.axes;
  const trainingX = useMemo(() => finiteValues(projection.training.x), [projection]);
  const syntheticX = useMemo(() => finiteValues(projection.synthetic.x), [projection]);
  const trainingY = useMemo(() => finiteValues(projection.training.y), [projection]);
  const syntheticY = useMemo(() => finiteValues(projection.synthetic.y), [projection]);
  return (
    <Stack gap="density-sm" data-testid="text-semantic-charts">
      <Text kind="label/bold/md">{column}</Text>
      <Grid cols={{ xs: 1, lg: 3 }} gap="density-lg">
        <ComponentHistogram label={xLabel} training={trainingX} synthetic={syntheticX} />
        <ScatterPanel
          series={[
            { dataset: 'training', points: training },
            { dataset: 'synthetic', points: synthetic },
          ]}
          xLabel={xLabel}
          yLabel={yLabel}
          xDomain={xDomain}
          yDomain={yDomain}
        />
        <ComponentHistogram label={yLabel} training={trainingY} synthetic={syntheticY} />
      </Grid>
    </Stack>
  );
};
