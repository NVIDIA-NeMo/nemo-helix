// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Table, Text } from '@nvidia/foundations-react-core';
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
import type { FC, ReactNode } from 'react';

export interface DistillationScoreboardProps {
  arms: MockArmResult[];
}

const TONE_CLASSES: Record<Delta['tone'], string> = {
  success: 'text-feedback-success',
  warning: 'text-feedback-warning',
  error: 'text-feedback-danger',
};

const usd = (value: number | null): string => (value === null ? '—' : `$${value.toFixed(2)}`);

/** All models side by side, one row per metric, each value with its change against the baseline. */
export const DistillationScoreboard: FC<DistillationScoreboardProps> = ({ arms }) => {
  const baseline = arms[0];

  const cell = (value: ReactNode, delta?: Delta): ReactNode => (
    <Stack gap="density-xs">
      <Text kind="body/semibold/md">{value}</Text>
      {delta && (
        <Text kind="body/regular/xs" className={TONE_CLASSES[delta.tone]}>
          {delta.text}
        </Text>
      )}
    </Stack>
  );

  const rows: { label: string; values: (arm: MockArmResult) => ReactNode }[] = [
    {
      label: 'Model',
      values: (arm) => <Text kind="mono/sm">{arm.model}</Text>,
    },
    {
      label: 'Score',
      values: (arm) => cell(arm.score.toFixed(2), arm.id === baseline.id ? undefined : scoreDelta(arm, baseline)),
    },
    {
      label: 'Est. cost per 1,000 tasks',
      values: (arm) =>
        cell(usd(arm.estimatedCostPer1kTasks), arm.id === baseline.id ? undefined : costDelta(arm, baseline)),
    },
    {
      label: 'Median latency',
      values: (arm) =>
        cell(`${arm.medianLatencySec.toFixed(1)} s`, arm.id === baseline.id ? undefined : latencyDelta(arm, baseline)),
    },
    {
      label: 'Tokens per task (in / out)',
      values: (arm) =>
        cell(`${arm.inputTokensPerTask.toLocaleString()} / ${arm.outputTokensPerTask.toLocaleString()}`),
    },
    {
      label: 'Evaluation',
      values: (arm) => <Text kind="mono/sm">{arm.evalJobName}</Text>,
    },
  ];

  return (
    <Stack gap="density-md" className="w-full">
      <Table
        data-testid="distillation-scoreboard"
        className="w-full bg-transparent"
        align="left"
        columns={[
          { children: 'Evaluation' },
          ...arms.map((arm) => ({
            children: (
              <Flex align="center" gap="density-sm">
                <span aria-hidden className={cn('size-3 shrink-0 rounded-full', ARM_COLOR_CLASSES[arm.id])} />
                <Text kind="body/semibold/sm">{arm.title}</Text>
              </Flex>
            ),
          })),
        ]}
        rows={rows.map((row) => ({
          cells: [
            { children: <Text kind="body/regular/sm" className="text-secondary">{row.label}</Text> },
            ...arms.map((arm) => ({ children: row.values(arm) })),
          ],
        }))}
      />
    </Stack>
  );
};
