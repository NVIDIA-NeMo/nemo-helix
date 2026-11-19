// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile, type StatTileStatus } from '@nemo/common/src/components/StatTile';
import {
  Badge,
  Flex,
  Grid,
  Panel,
  RadioGroupInput,
  RadioGroupItem,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  costDelta,
  type Delta,
  latencyDelta,
  scoreDelta,
} from '@studio/routes/agents/AgentDetailRoute/distillation/distillationDeltas';
import {
  armColorClass,
  type MockArmResult,
  ORIGINAL_ARM,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import cn from 'classnames';
import type { FC } from 'react';

export interface DistillationArmCardProps {
  arm: MockArmResult;
  /** The original agent's result, which every other arm's deltas are measured against. */
  baseline: MockArmResult;
  /** The best-scoring candidate. */
  best?: boolean;
  /** Makes the card a radio option; it must sit inside a `RadioGroupRoot`. */
  selectable?: boolean;
  /** The candidate Promote would use. */
  selected?: boolean;
  /** Show the score against the untrained student too, as for a candidate. */
  showUntrainedDelta?: boolean;
  /** What to call the untrained student's figures, when this arm is shown against them too. */
  untrained?: MockArmResult;
}

const TONE_CLASSES: Record<Delta['tone'], string> = {
  success: 'text-feedback-success',
  warning: 'text-feedback-warning',
  error: 'text-feedback-danger',
};

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
  best,
  selectable,
  selected,
  showUntrainedDelta,
  untrained,
}) => {
  const isBaseline = arm.id === baseline.id;
  const referenceName = baseline.id === ORIGINAL_ARM ? 'baseline' : 'untrained';
  const scoreTile = isBaseline ? undefined : toTileDelta(scoreDelta(arm, baseline, referenceName));
  const costTile = isBaseline ? undefined : toTileDelta(costDelta(arm, baseline, referenceName));
  const latencyTile = isBaseline ? undefined : toTileDelta(latencyDelta(arm, baseline, referenceName));
  const againstUntrained =
    showUntrainedDelta && untrained && untrained.id !== baseline.id
      ? scoreDelta(arm, untrained, 'untrained')
      : undefined;

  const labelId = `distillation-arm-${arm.id}-label`;

  const card = (
    <Panel
      slotHeading={
        <Flex align="center" gap="density-md">
          {selectable && <RadioGroupInput value={arm.id} aria-labelledby={labelId} />}
          <span
            aria-hidden
            className={cn('size-3 shrink-0 rounded-full', armColorClass(arm))}
          />
          <Text kind="title/xs" id={labelId}>
            {arm.title}
          </Text>
          {best && (
            <Badge kind="solid" color="green">
              Best score
            </Badge>
          )}
        </Flex>
      }
      elevation="high"
      density="compact"
      className={cn(
        'h-full',
        selectable && 'cursor-pointer hover:bg-interaction-hover [&_*]:cursor-pointer',
        selected && 'border-2 border-interaction-selected'
      )}
      data-testid={`distillation-arm-${arm.id}`}
    >
      <Stack gap="density-lg">
        <Stack gap="density-md">
          {[
            { label: 'Model', value: arm.model },
            { label: 'Evaluation', value: arm.evalJobName },
            {
              label: 'Input / output tokens per task',
              value: `${arm.inputTokensPerTask.toLocaleString()} / ${arm.outputTokensPerTask.toLocaleString()}`,
            },
          ].map((item) => (
            <Stack key={item.label} gap="density-xs">
              <Text kind="body/regular/xs" className="text-secondary">
                {item.label}
              </Text>
              <Text kind="body/regular/sm" className="min-w-0 break-all">
                {item.value}
              </Text>
            </Stack>
          ))}
        </Stack>
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
            value={arm.estimatedCostPer1kTasks === null ? '—' : `$${arm.estimatedCostPer1kTasks.toFixed(2)}`}
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
        {againstUntrained && (
          <Text kind="body/regular/xs" className={TONE_CLASSES[againstUntrained.tone]}>
            Score {againstUntrained.text}
          </Text>
        )}
      </Stack>
    </Panel>
  );

  return selectable ? (
    <RadioGroupItem aria-labelledby={labelId} className="block h-full w-full">
      {card}
    </RadioGroupItem>
  ) : (
    card
  );
};
