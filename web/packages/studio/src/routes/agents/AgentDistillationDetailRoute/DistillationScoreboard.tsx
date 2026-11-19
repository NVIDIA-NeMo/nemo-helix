// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Badge,
  Flex,
  RadioGroupInput,
  RadioGroupItem,
  Stack,
  Table,
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
  BASE_ARM,
  type MockArmResult,
  ORIGINAL_ARM,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import cn from 'classnames';
import type { FC, ReactNode } from 'react';

export interface DistillationScoreboardProps {
  arms: MockArmResult[];
  /** The candidate Promote would use. */
  selectedArm?: string;
  bestArm?: string;
  /** Candidate rows get a radio; the table must sit inside a `RadioGroupRoot`. */
  selectable?: boolean;
}

const TONE_CLASSES: Record<Delta['tone'], string> = {
  success: 'text-feedback-success',
  warning: 'text-feedback-warning',
  error: 'text-feedback-danger',
};

const usd = (value: number | null): string => (value === null ? '—' : `$${value.toFixed(2)}`);

const isCandidate = (arm: MockArmResult): boolean => arm.id !== ORIGINAL_ARM && arm.id !== BASE_ARM;

/**
 * The models as rows: the baseline and the untrained student for reference, then each candidate. A candidate's score is shown against the baseline and the untrained model.
 */
export const DistillationScoreboard: FC<DistillationScoreboardProps> = ({
  arms,
  selectedArm,
  bestArm,
  selectable,
}) => {
  const reference = arms[0];
  const untrained = arms.find((arm) => arm.id === BASE_ARM);
  const referenceName = reference.id === ORIGINAL_ARM ? 'baseline' : 'untrained';

  const cell = (value: ReactNode, ...deltas: (Delta | undefined)[]): ReactNode => (
    <Stack gap="density-xs">
      <Text kind="body/semibold/md">{value}</Text>
      {deltas.map(
        (delta) =>
          delta && (
            <Text key={delta.text} kind="body/regular/xs" className={TONE_CLASSES[delta.tone]}>
              {delta.text}
            </Text>
          )
      )}
    </Stack>
  );

  const row = (arm: MockArmResult) => {
    const comparedToReference = arm.id !== reference.id;
    const candidate = isCandidate(arm);
    return {
      cells: [
        {
          children:
            selectable && candidate ? (
              <RadioGroupItem aria-label={`Promote ${arm.title}`}>
                <RadioGroupInput value={arm.id} aria-label={`Promote ${arm.title}`} />
              </RadioGroupItem>
            ) : null,
        },
        {
          children: (
            <Stack
              gap="density-xs"
              data-testid={`distillation-arm-${arm.id}`}
              className={cn('rounded-md', selectedArm === arm.id && 'bg-interaction-selected')}
            >
              <Flex align="center" gap="density-sm">
                <span aria-hidden className={cn('size-3 shrink-0 rounded-full', armColorClass(arm))} />
                <Text kind="body/semibold/sm">{arm.title}</Text>
                {candidate && bestArm === arm.id && (
                  <Badge kind="solid" color="green">
                    Best score
                  </Badge>
                )}
              </Flex>
              <Text kind="mono/sm" className="text-secondary">
                {arm.model}
              </Text>
            </Stack>
          ),
        },
        {
          children: cell(
            arm.score.toFixed(2),
            comparedToReference ? scoreDelta(arm, reference, referenceName) : undefined,
            candidate && untrained && untrained !== reference ? scoreDelta(arm, untrained, 'untrained') : undefined
          ),
        },
        {
          children: cell(
            usd(arm.estimatedCostPer1kTasks),
            comparedToReference ? costDelta(arm, reference, referenceName) : undefined
          ),
        },
        {
          children: cell(
            `${arm.medianLatencySec.toFixed(1)} s`,
            comparedToReference ? latencyDelta(arm, reference, referenceName) : undefined
          ),
        },
        {
          children: cell(
            `${arm.inputTokensPerTask.toLocaleString()} / ${arm.outputTokensPerTask.toLocaleString()}`
          ),
        },
      ],
    };
  };

  return (
    <Stack className="w-full">
      <Table
        data-testid="distillation-scoreboard"
        className="w-full bg-transparent"
        align="left"
        columns={[
          { children: '' },
          { children: 'Model' },
          { children: 'Score' },
          { children: 'Est. cost / 1k tasks' },
          { children: 'Median latency' },
          { children: 'Tokens / task (in / out)' },
        ]}
        rows={arms.map(row)}
      />
    </Stack>
  );
};
