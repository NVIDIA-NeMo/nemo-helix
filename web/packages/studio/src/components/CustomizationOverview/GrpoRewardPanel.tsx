// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile, type StatTileDiagnosticProps } from '@nemo/common/src/components/StatTile';
import { Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { RangeBand, type RangeBandSeries } from '@studio/components/charts/RangeBand';
import type { RewardChartData } from '@studio/util/grpoMetrics';
import type { FC } from 'react';

interface Props {
  reward?: RewardChartData;
  metrics: StatTileDiagnosticProps[];
  progress: StatTileDiagnosticProps[];
}

const TRAINING_COLOR = 'var(--text-color-accent-green)';
const VALIDATION_COLOR = 'var(--text-color-accent-yellow)';

const EMPTY: RewardChartData = {
  steps: [],
  training: [],
  validation: [],
};

const NO_BAND: (number | null)[] = [];

/**
 * Replaces the loss chart the other backends show: GRPO's loss is a policy-gradient surrogate
 * whose magnitude means nothing, so reward is the only curve that says whether the run works.
 */
export const GrpoRewardPanel: FC<Props> = ({ reward = EMPTY, metrics, progress }) => {
  const series: RangeBandSeries[] = [
    {
      id: 'training',
      label: 'Training reward',
      data: reward.training,
      lower: NO_BAND,
      upper: NO_BAND,
      color: TRAINING_COLOR,
    },
    {
      id: 'validation',
      label: 'Validation reward',
      data: reward.validation,
      lower: NO_BAND,
      upper: NO_BAND,
      color: VALIDATION_COLOR,
      dashed: true,
      // Validation runs every N steps: bridge the gaps rather than invent points, and keep the
      // markers, which the chart-level default would drop for measuring the padded array.
      connectNulls: true,
      showMarks: true,
    },
  ];

  return (
    <Stack gap="density-xl">
      {metrics.length > 0 && (
        <Grid cols={{ base: 1, md: 2, lg: 4 }} gap="density-xl">
          {metrics.map((tile) => (
            // `StatTile` caps itself at `max-w-sm`; without tailwind-merge only `!` beats it.
            <StatTile key={tile.label} {...tile} className="max-w-none!" />
          ))}
        </Grid>
      )}

      <Panel elevation="high">
        <Stack gap="density-xl">
          <Flex align="start" justify="between" gap="density-2xl" wrap="wrap">
            <Stack gap="density-xs">
              <Text kind="label/bold/lg">Reward</Text>
              <Text kind="body/regular/sm" className="text-secondary">
                Mean reward per step across all sampled rollouts, with validation passes overlaid.
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
          <RangeBand
            series={series}
            xAxis={reward.steps}
            xAxisLabel="Step"
            yAxisLabel="Mean reward"
            emptyMessage="No reward data available"
          />
        </Stack>
      </Panel>
    </Stack>
  );
};
