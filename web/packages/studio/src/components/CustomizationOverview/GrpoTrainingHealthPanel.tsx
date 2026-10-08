// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { ComparisonLineChart } from '@nemo/common/src/components/ComparisonLineChart';
import { StatTile } from '@nemo/common/src/components/StatTile';
import { Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import type {
  CustomizationMetricValue,
  CustomizationStatusDetailsWithMetrics,
} from '@studio/types/customization';
import {
  GRPO_DIAGNOSTICS,
  readSeries,
  thresholdAxisBounds,
  type GrpoDiagnostic,
} from '@studio/util/grpoMetrics';
import type { FC } from 'react';

interface Props {
  statusDetails?: CustomizationStatusDetailsWithMetrics;
}

interface ReportedDiagnostic {
  diagnostic: GrpoDiagnostic;
  series: CustomizationMetricValue[];
}

const CHART_HEIGHT = 180;

/**
 * Collapsed by default — only worth the scroll once reward stalls. `AccordionPanel` unmounts its
 * children while closed, so the charts cost nothing until then.
 */
export const GrpoTrainingHealthPanel: FC<Props> = ({ statusDetails }) => {
  const reported: ReportedDiagnostic[] = GRPO_DIAGNOSTICS.map((diagnostic) => ({
    diagnostic,
    series: readSeries(statusDetails, diagnostic.metric) ?? [],
  })).filter((entry) => entry.series.length > 0);

  // Nothing behind the chevron beats a panel that opens onto an empty grid.
  if (reported.length === 0) {
    return null;
  }

  const tiles = reported.filter(({ diagnostic }) => diagnostic.tile);

  // A flat curve is dropped as well as an unwanted one: recharts pads a degenerate domain into an
  // invented y range, which reads as a broken axis rather than a flat line.
  const charts = reported.filter(
    ({ diagnostic, series }) =>
      diagnostic.chart && new Set(series.map((point) => point.value)).size > 1
  );

  return (
    <AccordionPanel slotHeading="Training health">
      <Stack gap="density-xl">
        <Text kind="body/regular/sm" className="text-secondary">
          Diagnostics from NeMo RL that predict divergence before reward does.
        </Text>

        {tiles.length > 0 && (
          <Grid cols={{ base: 1, md: 2, lg: 4 }} gap="density-xl">
            {tiles.map(({ diagnostic, series }) => {
              const verdict = diagnostic.evaluate?.(series);
              return (
                <StatTile
                  key={diagnostic.id}
                  label={diagnostic.metric}
                  value={diagnostic.formatValue(series[series.length - 1].value)}
                  trailingLabel={verdict?.label}
                  trailingLabelStatus={verdict?.status}
                  status={verdict?.status}
                  hint={diagnostic.hint}
                />
              );
            })}
          </Grid>
        )}

        {charts.length > 0 && (
          <Grid cols={{ base: 1, lg: 2 }} gap="density-xl">
            {charts.map(({ diagnostic, series }) => (
              <Stack key={diagnostic.id} gap="density-sm">
                <Flex justify="between" align="baseline" gap="density-md">
                  <Text kind="label/bold/md">{diagnostic.title}</Text>
                  <Text kind="body/regular/sm" className="text-secondary">
                    {diagnostic.metric}
                  </Text>
                </Flex>
                <ComparisonLineChart
                  series={[
                    {
                      id: diagnostic.id,
                      label: diagnostic.metric,
                      data: series.map((point) => point.value),
                      valueFormatter: (value) =>
                        value === null ? '—' : diagnostic.formatValue(value),
                    },
                  ]}
                  xAxis={series.map((point) => point.step)}
                  xAxisLabel="Step"
                  height={CHART_HEIGHT}
                  showLegend={false}
                  emptyMessage="No training metrics to compare"
                  referenceLines={diagnostic.referenceLines}
                  {...thresholdAxisBounds(diagnostic, series)}
                  formatYValue={diagnostic.formatAxisValue ?? diagnostic.formatValue}
                />
              </Stack>
            ))}
          </Grid>
        )}
      </Stack>
    </AccordionPanel>
  );
};
