// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile, type StatTileDiagnosticProps } from '@nemo/common/src/components/StatTile';
import { Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { TrainValidationLossLineChart } from '@studio/components/charts/TrainValidationLossLineChart';
import type { CustomizationMetricValue } from '@studio/types/customization';
import type { FC } from 'react';

interface Props {
  trainLoss?: CustomizationMetricValue[];
  valLoss?: CustomizationMetricValue[];
  maxSteps: number;
  metrics: StatTileDiagnosticProps[];
  progress: StatTileDiagnosticProps[];
}

export const TrainingLossPanel: FC<Props> = ({
  trainLoss,
  valLoss,
  maxSteps,
  metrics,
  progress,
}) => (
  <Panel elevation="high">
    <Stack gap="density-xl">
      <Flex gap="density-5xl" align="start" wrap="wrap">
        {metrics.length > 0 && (
          <Grid cols={2} gap="density-5xl" className="m-auto w-full min-w-0 shrink-0 lg:w-[22rem]">
            {metrics.map((tile) => (
              <StatTile key={tile.label} {...tile} bordered={false} />
            ))}
          </Grid>
        )}
        <Stack className="min-w-0 flex-1" gap="density-xl">
          <Flex align="start" justify="between" gap="density-2xl" wrap="wrap">
            <Stack gap="density-xs">
              <Text kind="label/bold/lg">Training loss</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                Loss per step across the training run, with validation passes overlaid.
              </Text>
            </Stack>
            {progress.length > 0 && (
              <Flex gap="density-2xl" wrap="wrap">
                {progress.map((tile) => (
                  <StatTile key={tile.label} {...tile} bordered={false} />
                ))}
              </Flex>
            )}
          </Flex>
          <TrainValidationLossLineChart
            trainLoss={trainLoss}
            valLoss={valLoss}
            attributes={{ XAxis: { domain: ['dataMin', maxSteps] } }}
          />
        </Stack>
      </Flex>
    </Stack>
  </Panel>
);
