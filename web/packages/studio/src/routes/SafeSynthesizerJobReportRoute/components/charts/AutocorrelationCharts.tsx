// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ChartTooltipSurface,
  chartTooltipProps,
} from '@nemo/common/src/components/charts/ChartTooltip';
import {
  GRID_PROPS,
  TICK_STYLE,
  xAxisLabelProps,
  yAxisLabelProps,
} from '@nemo/common/src/components/charts/frame';
import { REFERENCE_LINE_COLOR } from '@nemo/common/src/components/charts/tokens';
import { Grid, Text } from '@nvidia/foundations-react-core';
import {
  autocorrelationSummaryRows,
  DATASET_COLORS,
  HEAT_COLORS,
  heatColor,
  lagDifferenceRows,
  orderAutocorrelationColumns,
  seededRandom,
  shortenLabel,
  groupLabel,
} from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import {
  ChartFigure,
  ColumnSelect,
} from '@studio/routes/SafeSynthesizerJobReportRoute/components/charts/ChartFigure';
import { CHART_TOOLTIPS } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type { AutocorrelationProfile } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { PAIR_SCORE_MAX_COLUMNS } from '@studio/routes/SafeSynthesizerJobReportRoute/timeSeriesSummary';
import { type FC, useMemo, useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const CHART_HEIGHT = 300;
const MARGIN = { top: 16, right: 16, bottom: 24, left: 8 };
const BAND_OPACITY = 0.2;
const LAG_DIFFERENCE_COLOR_MAX = 0.5;
const PAIR_SCORE_JITTER = 0.25;

const formatAcf = (value: number) => value.toFixed(3);

const TypicalAutocorrelation: FC<{ profiles: AutocorrelationProfile[]; columns: string[] }> = ({
  profiles,
  columns,
}) => {
  const [column, setColumn] = useState(columns[0]);
  const rows = useMemo(() => autocorrelationSummaryRows(profiles, column), [profiles, column]);
  return (
    <ChartFigure
      title="Typical autocorrelation"
      tooltip={CHART_TOOLTIPS.autocorrelationTypical}
      slotEnd={<ColumnSelect columns={columns} value={column} onChange={setColumn} />}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <ComposedChart data={rows} margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="lag" tick={TICK_STYLE} label={xAxisLabelProps('Lag')} />
          <YAxis
            domain={[-1.05, 1.05]}
            ticks={[-1, -0.5, 0, 0.5, 1]}
            tick={TICK_STYLE}
            width={48}
            label={yAxisLabelProps('Autocorrelation')}
          />
          <Tooltip
            {...chartTooltipProps}
            labelFormatter={(lag) => `lag ${lag}`}
            formatter={(value: number | [number, number]) =>
              Array.isArray(value) ? value.map(formatAcf).join(' to ') : formatAcf(value)
            }
          />
          <Area
            dataKey="trainingBand"
            name="Training interquartile range"
            stroke="none"
            fill={DATASET_COLORS.training}
            fillOpacity={BAND_OPACITY}
            isAnimationActive={false}
          />
          <Area
            dataKey="syntheticBand"
            name="Synthetic interquartile range"
            stroke="none"
            fill={DATASET_COLORS.synthetic}
            fillOpacity={BAND_OPACITY}
            isAnimationActive={false}
          />
          <Line
            dataKey="trainingMedian"
            name="Training median ACF"
            stroke={DATASET_COLORS.training}
            dot={{ r: 2 }}
            isAnimationActive={false}
          />
          <Line
            dataKey="syntheticMedian"
            name="Synthetic median ACF"
            stroke={DATASET_COLORS.synthetic}
            dot={{ r: 2 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

const DifferenceByLag: FC<{ profiles: AutocorrelationProfile[] }> = ({ profiles }) => {
  const rows = useMemo(() => lagDifferenceRows(profiles), [profiles]);
  const top = Math.max(1, ...rows.map((row) => row.difference * 1.05));
  return (
    <ChartFigure title="Difference by lag" tooltip={CHART_TOOLTIPS.autocorrelationDifference}>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <BarChart data={rows} margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="lag" tick={TICK_STYLE} label={xAxisLabelProps('Lag')} />
          <YAxis
            domain={[0, top]}
            tick={TICK_STYLE}
            tickFormatter={(value: number) => value.toFixed(1)}
            width={48}
            label={yAxisLabelProps('Mean |ACF difference|')}
          />
          <Tooltip
            {...chartTooltipProps}
            cursor={{ fill: 'var(--background-color-accent-gray-subtle)' }}
            labelFormatter={(lag) => `lag ${lag}`}
            formatter={(value: number, _name, item) => [
              `${formatAcf(value)} (${item.payload.pairs} pairs)`,
              'Mean absolute difference',
            ]}
          />
          <Bar
            dataKey="difference"
            stroke={REFERENCE_LINE_COLOR}
            strokeWidth={0.5}
            isAnimationActive={false}
          >
            {rows.map((row) => (
              <Cell
                key={row.lag}
                fill={heatColor(HEAT_COLORS.difference, row.difference / LAG_DIFFERENCE_COLOR_MAX)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

interface PairDot {
  x: number;
  y: number;
  column: string;
  group: string;
  similarity: number;
}

const pairDots = (profiles: AutocorrelationProfile[], columns: string[]) => {
  const rows = [...columns].reverse();
  const random = seededRandom(0);
  return columns.flatMap((column) =>
    profiles
      .filter((profile) => profile.column === column)
      .map<PairDot>((profile) => ({
        x: 10 * profile.similarity,
        y: rows.indexOf(column) + (random() * 2 - 1) * PAIR_SCORE_JITTER,
        column,
        group: groupLabel(profile.group),
        similarity: profile.similarity,
      }))
  );
};

const PairScores: FC<{
  profiles: AutocorrelationProfile[];
  columns: string[];
  overallScore: number | null;
}> = ({ profiles, columns, overallScore }) => {
  const plotted = useMemo(() => columns.slice(0, PAIR_SCORE_MAX_COLUMNS), [columns]);
  const rows = useMemo(() => [...plotted].reverse(), [plotted]);
  const dots = useMemo(() => pairDots(profiles, plotted), [profiles, plotted]);
  return (
    <ChartFigure title="Pair scores" tooltip={CHART_TOOLTIPS.autocorrelationPairScores}>
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <ScatterChart margin={MARGIN}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis
            type="number"
            dataKey="x"
            domain={[-0.25, 10.25]}
            ticks={[0, 2, 4, 6, 8, 10]}
            tick={TICK_STYLE}
            label={xAxisLabelProps('Pair score')}
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.6, rows.length - 0.4]}
            ticks={rows.map((_, index) => index)}
            tickFormatter={(index: number) => shortenLabel(rows[index] ?? '')}
            tick={TICK_STYLE}
            width={80}
          />
          <Tooltip
            {...chartTooltipProps}
            content={({ payload }) => {
              const dot = payload?.[0]?.payload as PairDot | undefined;
              if (!dot) {
                return null;
              }
              return (
                <ChartTooltipSurface label={dot.column}>
                  <Text kind="body/regular/sm">group {dot.group}</Text>
                  <Text kind="body/regular/sm">score {dot.x.toFixed(1)}</Text>
                </ChartTooltipSurface>
              );
            }}
          />
          {overallScore !== null ? (
            <ReferenceLine
              x={overallScore}
              stroke={REFERENCE_LINE_COLOR}
              strokeDasharray="4 4"
              label={{
                value: `score ${overallScore.toFixed(1)}`,
                position: 'top',
                style: TICK_STYLE,
              }}
            />
          ) : null}
          <Scatter data={dots} isAnimationActive={false}>
            {dots.map((dot, index) => (
              <Cell
                key={`${dot.column}-${dot.group}-${index}`}
                fill={heatColor(HEAT_COLORS.difference, 1 - dot.similarity)}
                stroke={REFERENCE_LINE_COLOR}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFigure>
  );
};

/** Typical profiles, mean difference by lag, and per-pair scores, as in the library report. */
export const AutocorrelationCharts: FC<{
  profiles: AutocorrelationProfile[];
  overallScore: number | null;
}> = ({ profiles, overallScore }) => {
  const columns = useMemo(() => orderAutocorrelationColumns(profiles), [profiles]);
  return (
    <Grid cols={{ xs: 1, xl: 3 }} gap="density-lg" data-testid="autocorrelation-charts">
      <TypicalAutocorrelation profiles={profiles} columns={columns} />
      <DifferenceByLag profiles={profiles} />
      <PairScores profiles={profiles} columns={columns} overallScore={overallScore} />
    </Grid>
  );
};
