// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import {
  ARM_COLOR_CLASSES,
  ARM_COLORS,
  type MockArmResult,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import cn from 'classnames';
import type { FC } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface DistillationRelativeBarsProps {
  arms: MockArmResult[];
}

const percentOf = (value: number | null, reference: number | null): number | undefined =>
  value === null || reference === null ? undefined : Math.round((value / reference) * 100);

/** Score, cost and latency of each model as a percentage of the original agent. */
export const DistillationRelativeBars: FC<DistillationRelativeBarsProps> = ({ arms }) => {
  const original = arms[0];
  const data = [
    {
      metric: 'Score',
      ...Object.fromEntries(arms.map((arm) => [arm.id, percentOf(arm.score, original.score)])),
    },
    {
      metric: 'Cost',
      ...Object.fromEntries(
        arms.map((arm) => [
          arm.id,
          percentOf(arm.estimatedCostPer1kTasks, original.estimatedCostPer1kTasks),
        ])
      ),
    },
    {
      metric: 'Latency',
      ...Object.fromEntries(
        arms.map((arm) => [arm.id, percentOf(arm.medianLatencySec, original.medianLatencySec)])
      ),
    },
  ];

  return (
    <Panel
      slotHeading="Relative to the baseline"
      elevation="high"
      density="compact"
      className="h-full"
    >
      <Stack gap="density-lg">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 40, bottom: 8, left: 8 }}
            barCategoryGap="22%"
          >
            <CartesianGrid stroke="var(--border-color-base)" strokeDasharray="3 3" horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 120]}
              ticks={[0, 25, 50, 75, 100]}
              tickFormatter={(value: number) => `${value}%`}
              tick={{ fontSize: 11 }}
            />
            <YAxis type="category" dataKey="metric" width={64} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(value: number) => `${value}% of baseline`} />
            <ReferenceLine x={100} stroke="var(--text-color-secondary)" strokeDasharray="4 4" />
            {arms.map((arm) => (
              <Bar key={arm.id} dataKey={arm.id} name={arm.title} fill={ARM_COLORS[arm.id]}>
                <LabelList
                  dataKey={arm.id}
                  position="right"
                  formatter={(value: number) => `${value}%`}
                  fontSize={11}
                />
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
        <Flex gap="density-xl" wrap="wrap" align="center">
          {arms.map((arm) => (
            <Flex key={arm.id} align="center" gap="density-sm">
              <span aria-hidden className={cn('size-3 rounded-full', ARM_COLOR_CLASSES[arm.id])} />
              <Text kind="body/regular/xs">{arm.title}</Text>
            </Flex>
          ))}
          <Text kind="body/regular/xs" className="text-secondary">
            Higher is better for score; lower is better for cost and latency.
          </Text>
        </Flex>
      </Stack>
    </Panel>
  );
};
