// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { StatTile, type StatTileStatus } from '@nemo/common/src/components/StatTile';
import { Badge, Flex, Grid, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  costDelta,
  type Delta,
  latencyDelta,
  scoreDelta,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationDeltas';
import {
  ARM_COLOR_CLASSES,
  type MockArmResult,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import cn from 'classnames';
import type { FC } from 'react';

export interface DistillationArmCardProps {
  arm: MockArmResult;
  /** The original agent's result, which every other arm's deltas are measured against. */
  baseline: MockArmResult;
  highlighted?: boolean;
}

interface TileDelta {
  hint: string;
  hintStatus: StatTileStatus;
  status?: StatTileStatus;
}

const toTileDelta = (delta: Delta | undefined): TileDelta | undefined =>
  delta && {
    hint: delta.text,
    hintStatus: delta.tone,
    ...(delta.tone === 'success' ? {} : { status: delta.tone }),
  };

/** One arm of the three-way comparison: its model and four headline numbers as stat tiles. */
export const DistillationArmCard: FC<DistillationArmCardProps> = ({
  arm,
  baseline,
  highlighted,
}) => {
  const isBaseline = arm.id === baseline.id;
  const scoreTile = isBaseline ? undefined : toTileDelta(scoreDelta(arm, baseline));
  const costTile = isBaseline ? undefined : toTileDelta(costDelta(arm, baseline));
  const latencyTile = isBaseline ? undefined : toTileDelta(latencyDelta(arm, baseline));

  return (
    <Panel
      slotHeading={
        <Flex align="center" gap="density-md">
          <span
            aria-hidden
            className={cn('size-3 shrink-0 rounded-full', ARM_COLOR_CLASSES[arm.id])}
          />
          <Text kind="title/xs">{arm.title}</Text>
          {highlighted && (
            <Badge kind="solid" color="green">
              Best value
            </Badge>
          )}
        </Flex>
      }
      elevation="high"
      density="compact"
      className={cn(
        'h-full',
        highlighted && 'border-2 border-(--color-purple-500)'
      )}
      data-testid={`distillation-arm-${arm.id}`}
    >
      <Stack gap="density-lg">
        <KeyValueGrid
          items={[
            { key: 'model', label: 'Model', value: arm.model, wrapValue: true },
            { key: 'eval', label: 'Evaluation', value: arm.evalJobName, wrapValue: true },
            {
              key: 'tokens',
              label: 'Input / output tokens per task',
              value: `${arm.inputTokensPerTask.toLocaleString()} / ${arm.outputTokensPerTask.toLocaleString()}`,
            },
          ]}
        />
        <Grid cols={2} gap="density-lg">
          <StatTile
            variant="metric"
            label="Score"
            value={arm.score.toFixed(2)}
            trailingLabel="/ 1.00"
            {...(isBaseline ? { hint: 'Baseline' } : scoreTile)}
          />
          <StatTile
            variant="metric"
            label="Cost / 1k tasks"
            value={`$${arm.estimatedCostPer1kTasks.toFixed(2)}`}
            trailingLabel="est."
            {...costTile}
          />
          <StatTile
            variant="metric"
            label="Median latency"
            value={arm.medianLatencySec.toFixed(1)}
            trailingLabel="s"
            {...latencyTile}
          />
          <StatTile
            variant="metric"
            label="Tokens / task"
            value={(arm.inputTokensPerTask + arm.outputTokensPerTask).toLocaleString()}
          />
        </Grid>
      </Stack>
    </Panel>
  );
};
