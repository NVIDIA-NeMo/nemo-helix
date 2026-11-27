// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { chartTooltipProps } from '@nemo/common/src/components/charts/ChartTooltip';
import {
  GRID_PROPS,
  TICK_STYLE,
  xAxisLabelProps,
  yAxisLabelProps,
} from '@nemo/common/src/components/charts/frame';
import { REFERENCE_LINE_COLOR } from '@nemo/common/src/components/charts/tokens';
import { Grid } from '@nvidia/foundations-react-core';
import {
  DATASET_COLORS,
  finiteValues,
  groupLabel,
  HEAT_COLORS,
  heatColor,
  histogram,
  quantile,
  shortenLabel,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import {
  ChartFigure,
  ColumnSelect,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/ChartFigure';
import { CHART_TOOLTIPS } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type { WindowMembershipInferenceDetails as Details } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type FC, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const CHART_HEIGHT = 340;
const MARGIN = { top: 16, right: 16, bottom: 24, left: 8 };
const DISTANCE_BINS = 30;
const DISTANCE_RANGE_QUANTILE = 0.99;
const AREA_OPACITY = 0.45;

const windowValueCount = (details: Details): number =>
  (details.window_size ?? 0) * details.columns.length;

const distanceRows = (details: Details) => {
  const scale = Math.sqrt(windowValueCount(details)) || 1;
  const training = finiteValues(details.distances?.training ?? []).map((value) => value / scale);
  const holdout = finiteValues(details.distances?.holdout ?? []).map((value) => value / scale);
  const threshold =
    details.attack?.threshold !== null && details.attack?.threshold !== undefined
      ? details.attack.threshold / scale
      : null;
  const limits = [training, holdout]
    .filter((values) => values.length > 0)
    .map((values) =>
      quantile(
        [...values].sort((a, b) => a - b),
        DISTANCE_RANGE_QUANTILE
      )
    );
  if (threshold !== null) {
    limits.push(threshold);
  }
  const top = Math.max(...limits, 1e-6) * 1.05;
  const edges = Array.from(
    { length: DISTANCE_BINS + 1 },
    (_, index) => (top * index) / DISTANCE_BINS
  );
  const trainingBins = histogram(training, edges, 'percent');
  const holdoutBins = histogram(holdout, edges, 'percent');
  const rows = trainingBins.map((value, index) => ({
    edge: edges[index],
    training: value,
    holdout: holdout.length ? holdoutBins[index] : undefined,
  }));
  return { rows: [...rows, { ...rows[rows.length - 1], edge: top }], threshold, top };
};

const DistanceToSynthetic: FC<{ details: Details }> = ({ details }) => {
  const { rows, threshold, top } = useMemo(() => distanceRows(details), [details]);
  return (
    <ChartFigure title="Distance to synthetic data" tooltip={CHART_TOOLTIPS.windowDistance}>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <AreaChart data={rows} margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis
            dataKey="edge"
            type="number"
            domain={[0, top]}
            tick={TICK_STYLE}
            tickFormatter={(value: number) => value.toFixed(2)}
            label={xAxisLabelProps('Difference per value (training sd)')}
          />
          <YAxis
            tick={TICK_STYLE}
            width={48}
            unit="%"
            label={yAxisLabelProps('Share of windows (%)')}
          />
          <Tooltip
            {...chartTooltipProps}
            labelFormatter={(edge: number) => `difference ${edge.toFixed(2)}`}
            formatter={(value: number) => `${value.toFixed(1)}% of windows`}
          />
          <Area
            type="stepAfter"
            dataKey="training"
            name="Training windows"
            stroke={DATASET_COLORS.training}
            fill={DATASET_COLORS.training}
            fillOpacity={AREA_OPACITY}
            isAnimationActive={false}
          />
          <Area
            type="stepAfter"
            dataKey="holdout"
            name="Holdout windows"
            stroke={DATASET_COLORS.holdout}
            fill={DATASET_COLORS.holdout}
            fillOpacity={AREA_OPACITY}
            isAnimationActive={false}
          />
          {threshold !== null ? (
            <ReferenceLine
              x={threshold}
              stroke={REFERENCE_LINE_COLOR}
              strokeDasharray="4 4"
              label={{ value: 'flag threshold', position: 'top', style: TICK_STYLE }}
            />
          ) : null}
        </AreaChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

const GroupsFlaggedMost: FC<{ details: Details; holdoutFlagRate: number }> = ({
  details,
  holdoutFlagRate,
}) => {
  const rows = useMemo(
    () =>
      details.per_group.flatMap((group) => {
        if (group.flag_rate === null) {
          return [];
        }
        const excess = (group.flag_rate - holdoutFlagRate) / Math.max(1 - holdoutFlagRate, 1e-12);
        return [
          {
            label: groupLabel(group.group),
            flagged: 100 * group.flag_rate,
            windows: group.windows,
            exactCopies: 100 * group.exact_copy_rate,
            excess,
          },
        ];
      }),
    [details, holdoutFlagRate]
  );
  return (
    <ChartFigure title="Groups flagged most" tooltip={CHART_TOOLTIPS.windowGroups}>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <BarChart data={rows} layout="vertical" margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} horizontal={false} vertical />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={TICK_STYLE}
            unit="%"
            label={xAxisLabelProps('Windows flagged (%)')}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={TICK_STYLE}
            tickFormatter={shortenLabel}
            width={80}
          />
          <Tooltip
            {...chartTooltipProps}
            cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
            formatter={(value: number, _name, item) => [
              `${value.toFixed(1)}% of ${item.payload.windows} windows flagged, ${item.payload.exactCopies.toFixed(1)}% exact copies`,
              'Flagged share',
            ]}
          />
          <ReferenceLine
            x={100 * holdoutFlagRate}
            stroke={REFERENCE_LINE_COLOR}
            strokeDasharray="4 4"
            label={{ value: 'holdout', position: 'top', style: TICK_STYLE }}
          />
          <Bar dataKey="flagged" stroke={REFERENCE_LINE_COLOR} isAnimationActive={false}>
            {rows.map((row) => (
              <Cell key={row.label} fill={heatColor(HEAT_COLORS.difference, row.excess)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

const ClosestMatch: FC<{ details: Details }> = ({ details }) => {
  const [column, setColumn] = useState(details.columns[0] ?? '');
  const match = details.closest_match;
  const columnIndex = Math.max(details.columns.indexOf(column), 0);
  const rows = useMemo(
    () =>
      (match?.synthetic_values[columnIndex] ?? []).map((synthetic, step) => ({
        step: step + 1,
        synthetic,
        training: match?.training_values?.[columnIndex]?.[step],
      })),
    [match, columnIndex]
  );
  return (
    <ChartFigure
      title="Closest match"
      tooltip={CHART_TOOLTIPS.windowClosest}
      slotEnd={<ColumnSelect columns={details.columns} value={column} onChange={setColumn} />}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={rows} margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={TICK_STYLE} label={xAxisLabelProps('Step in window')} />
          <YAxis tick={TICK_STYLE} width={48} label={yAxisLabelProps('Value')} />
          <Tooltip {...chartTooltipProps} labelFormatter={(step) => `step ${step}`} />
          {match?.training_values ? (
            <Line
              dataKey="training"
              name="Training window"
              stroke={DATASET_COLORS.training}
              isAnimationActive={false}
            />
          ) : null}
          <Line
            dataKey="synthetic"
            name="Synthetic window"
            stroke={DATASET_COLORS.synthetic}
            strokeDasharray="6 4"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

/** Window distances, most-flagged groups, and the closest synthetic window, as in the library report. */
export const WindowMembershipCharts: FC<{ details: Details }> = ({ details }) => {
  const hasDistances = Boolean(details.distances?.training.length);
  return (
    <Grid cols={{ xs: 1, xl: 3 }} gap="density-lg" data-testid="window-mia-charts">
      {hasDistances ? <DistanceToSynthetic details={details} /> : null}
      {details.attack !== null ? (
        <GroupsFlaggedMost details={details} holdoutFlagRate={details.attack.holdout_flag_rate} />
      ) : null}
      {details.closest_match !== null ? <ClosestMatch details={details} /> : null}
    </Grid>
  );
};
