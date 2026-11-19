// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  HELD_OUT_TASKS,
  type MockArmResult,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import { winTieLoss } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationTasks';
import cn from 'classnames';
import type { FC } from 'react';
import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface DistillationWinTieLossProps {
  arms: MockArmResult[];
}

const SERIES = [
  { key: 'wins', label: 'Better than baseline', color: 'var(--text-color-feedback-success)', className: 'bg-(--text-color-feedback-success)' },
  { key: 'ties', label: 'Same as baseline', color: 'var(--text-color-secondary)', className: 'bg-(--text-color-secondary)' },
  { key: 'losses', label: 'Worse than baseline', color: 'var(--text-color-feedback-danger)', className: 'bg-(--text-color-feedback-danger)' },
] as const;

/** For each challenger, on how many held-out tasks it beat, matched or lost to the original. */
export const DistillationWinTieLoss: FC<DistillationWinTieLossProps> = ({ arms }) => {
  const data = arms
    .filter((arm) => arm.id !== 'original')
    .map((arm) => ({ name: arm.title, ...winTieLoss(arm.id) }));

  return (
    <Panel
      slotHeading="Task by task vs. the baseline"
      elevation="high"
      density="compact"
      className="h-full"
    >
      <Stack gap="density-lg">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <XAxis
              type="number"
              domain={[0, HELD_OUT_TASKS]}
              tick={{ fontSize: 11 }}
              label={{ value: 'Held-out tasks', position: 'insideBottom', offset: -4 }}
            />
            <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
            <Tooltip />
            {SERIES.map((series) => (
              <Bar
                key={series.key}
                dataKey={series.key}
                name={series.label}
                stackId="outcome"
                fill={series.color}
              >
                <LabelList
                  dataKey={series.key}
                  position="center"
                  formatter={(value: number) => (value > 0 ? value : '')}
                  fontSize={12}
                />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
        <Flex gap="density-xl" wrap="wrap" align="center">
          {SERIES.map((series) => (
            <Flex key={series.key} align="center" gap="density-sm">
              <span aria-hidden className={cn('size-3 rounded-sm', series.className)} />
              <Text kind="body/regular/xs">{series.label}</Text>
            </Flex>
          ))}
        </Flex>
      </Stack>
    </Panel>
  );
};
