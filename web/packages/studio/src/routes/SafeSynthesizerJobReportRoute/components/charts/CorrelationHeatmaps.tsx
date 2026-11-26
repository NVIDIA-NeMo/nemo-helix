// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import { HEAT_COLORS, heatColor } from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import type { CorrelationMatrices } from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import type { FC } from 'react';

const CELL_GAP = 1;
const VIEWBOX = 100;

const formatCorrelation = (value: number | null | undefined): string =>
  value === null || value === undefined ? 'n/a' : value.toFixed(2);

interface HeatmapProps {
  title: string;
  color: string;
  values: (number | null)[][];
  matrices: CorrelationMatrices;
}

const Heatmap: FC<HeatmapProps> = ({ title, color, values, matrices }) => {
  const size = matrices.columns.length;
  const cell = VIEWBOX / Math.max(size, 1);
  return (
    <Stack gap="density-sm" data-testid="correlation-heatmap">
      <Text kind="label/bold/md">{title}</Text>
      <svg
        viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
        role="img"
        aria-label={title}
        className="aspect-square w-full max-w-[320px]"
      >
        {values.map((row, y) =>
          row.map((value, x) => (
            <rect
              key={`${matrices.columns[y]}-${matrices.columns[x]}`}
              x={x * cell + CELL_GAP / 2}
              y={y * cell + CELL_GAP / 2}
              width={cell - CELL_GAP}
              height={cell - CELL_GAP}
              fill={
                value === null
                  ? 'var(--background-color-accent-gray-subtle)'
                  : heatColor(color, value)
              }
            >
              <title>
                {`x: ${matrices.columns[x]}\ny: ${matrices.columns[y]}\nTraining correlation: ${formatCorrelation(matrices.training[y]?.[x])}\nSynthetic correlation: ${formatCorrelation(matrices.synthetic[y]?.[x])}\nAbsolute correlation difference: ${formatCorrelation(matrices.difference[y]?.[x])}`}
              </title>
            </rect>
          ))
        )}
      </svg>
      <Flex align="center" gap="density-sm">
        <Text kind="body/regular/sm" className="text-secondary">
          0
        </Text>
        <span
          className="h-2 w-24 rounded-sm"
          /* eslint-disable-next-line no-restricted-syntax */
          style={{ background: `linear-gradient(to right, ${heatColor(color, 0)}, ${color})` }}
        />
        <Text kind="body/regular/sm" className="text-secondary">
          1
        </Text>
      </Flex>
    </Stack>
  );
};

/** Training, synthetic, and absolute difference correlation heatmaps; hover a cell for its values. */
export const CorrelationHeatmaps: FC<{ matrices: CorrelationMatrices }> = ({ matrices }) => (
  <Grid cols={{ xs: 1, md: 3 }} gap="density-lg">
    <Heatmap
      title="Training Correlations"
      color={HEAT_COLORS.training}
      values={matrices.training}
      matrices={matrices}
    />
    <Heatmap
      title="Synthetic Correlations"
      color={HEAT_COLORS.synthetic}
      values={matrices.synthetic}
      matrices={matrices}
    />
    <Heatmap
      title="Correlation Difference (Absolute)"
      color={HEAT_COLORS.difference}
      values={matrices.difference}
      matrices={matrices}
    />
  </Grid>
);
