// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ArrowDown, ArrowUp } from '@nemo/icons/icons';
import { Badge, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { MockArmResult } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import type { FC } from 'react';

export interface DistillationArmCardProps {
  arm: MockArmResult;
  /** The original agent's result, which every other arm's deltas are measured against. */
  baseline: MockArmResult;
  highlighted?: boolean;
}

const percentChange = (value: number, reference: number): number =>
  Math.round(((value - reference) / reference) * 100);

const formatUsd = (value: number): string => `$${value.toFixed(2)}`;

/** One arm of the three-way comparison: its model, score, estimated cost and latency. */
export const DistillationArmCard: FC<DistillationArmCardProps> = ({
  arm,
  baseline,
  highlighted,
}) => {
  const isBaseline = arm.id === baseline.id;

  const renderDelta = (change: number, label: string, higherIsBetter: boolean) => {
    if (change === 0) return null;
    const isGood = higherIsBetter ? change > 0 : change < 0;
    const Arrow = change > 0 ? ArrowUp : ArrowDown;
    return (
      <Flex
        align="center"
        gap="density-xs"
        className={isGood ? 'text-feedback-success' : 'text-feedback-danger'}
      >
        <Arrow size="12" aria-hidden />
        <Text kind="body/semibold/xs">{label}</Text>
        <span className="sr-only">{isGood ? 'better' : 'worse'} than original</span>
      </Flex>
    );
  };

  const rows = [
    {
      label: 'Est. cost per 1,000 tasks',
      value: formatUsd(arm.estimatedCostPer1kTasks),
      delta: isBaseline
        ? null
        : renderDelta(
            percentChange(arm.estimatedCostPer1kTasks, baseline.estimatedCostPer1kTasks),
            `${Math.abs(percentChange(arm.estimatedCostPer1kTasks, baseline.estimatedCostPer1kTasks))}%`,
            false
          ),
    },
    {
      label: 'Median latency',
      value: `${arm.medianLatencySec.toFixed(1)} s`,
      delta: isBaseline
        ? null
        : renderDelta(
            percentChange(arm.medianLatencySec, baseline.medianLatencySec),
            `${Math.abs(percentChange(arm.medianLatencySec, baseline.medianLatencySec))}%`,
            false
          ),
    },
    {
      label: 'Tokens per task (in / out)',
      value: `${arm.inputTokensPerTask.toLocaleString()} / ${arm.outputTokensPerTask.toLocaleString()}`,
    },
  ];

  return (
    <Panel
      slotHeading={
        <Flex align="center" gap="density-md">
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
      data-testid={`distillation-arm-${arm.id}`}
    >
      <Stack gap="density-lg">
        <Text kind="mono/sm" className="text-secondary">
          {arm.model}
        </Text>
        <Stack gap="density-xs">
          <Text kind="body/regular/xs" className="text-secondary">
            Score on held-out tasks
          </Text>
          <Flex align="baseline" gap="density-md">
            <Text kind="title/lg">{arm.score.toFixed(2)}</Text>
            {!isBaseline &&
              renderDelta(
                Math.sign(arm.score - baseline.score),
                `${Math.abs(arm.score - baseline.score).toFixed(2)} vs original`,
                true
              )}
          </Flex>
        </Stack>
        {rows.map((row) => (
          <Flex key={row.label} justify="between" align="center" gap="density-md">
            <Text kind="body/regular/sm" className="text-secondary">
              {row.label}
            </Text>
            <Flex align="center" gap="density-md">
              <Text kind="body/semibold/sm">{row.value}</Text>
              {row.delta}
            </Flex>
          </Flex>
        ))}
        <Text kind="body/regular/xs" className="text-secondary">
          Evaluation job <span className="font-mono">{arm.evalJobName}</span>
        </Text>
      </Stack>
    </Panel>
  );
};
