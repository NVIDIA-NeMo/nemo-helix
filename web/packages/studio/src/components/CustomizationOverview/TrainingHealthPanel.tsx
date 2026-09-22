// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccordionPanel } from '@nemo/common/src/components/AccordionPanel';
import { ComparisonLineChart } from '@nemo/common/src/components/ComparisonLineChart';
import { Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import type {
  CustomizationMetricValue,
  CustomizationStatusDetailsWithMetrics,
} from '@studio/types/customization';
import { readSeries } from '@studio/util/grpoMetrics';
import { TRAINING_DIAGNOSTICS, type TrainingDiagnostic } from '@studio/util/trainingMetrics';
import type { FC } from 'react';

interface Props {
  statusDetails?: CustomizationStatusDetailsWithMetrics;
}

interface ReportedDiagnostic {
  diagnostic: TrainingDiagnostic;
  series: CustomizationMetricValue[];
}

const CHART_HEIGHT = 180;

/** Collapsed so the loss stays what the page opens on; AccordionPanel unmounts the charts until then. */
export const TrainingHealthPanel: FC<Props> = ({ statusDetails }) => {
  const reported: ReportedDiagnostic[] = TRAINING_DIAGNOSTICS.map((diagnostic) => ({
    diagnostic,
    series: readSeries(statusDetails, diagnostic.metric) ?? [],
  })).filter(({ series }) => series.length > 0);

  if (reported.length === 0) {
    return null;
  }

  return (
    <AccordionPanel slotHeading="Training health">
      <Stack gap="density-xl">
        <Text kind="body/regular/sm" className="text-secondary">
          The other series this run recorded, alongside its loss.
        </Text>

        <Grid cols={{ base: 1, lg: 2 }} gap="density-xl">
          {reported.map(({ diagnostic, series }) => (
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
                formatYValue={diagnostic.formatValue}
              />
            </Stack>
          ))}
        </Grid>
      </Stack>
    </AccordionPanel>
  );
};
