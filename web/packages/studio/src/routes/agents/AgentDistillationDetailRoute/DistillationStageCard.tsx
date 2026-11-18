// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { Flex, Panel, ProgressBar, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import type { MockStage } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import type { FC } from 'react';

export interface DistillationStageCardProps {
  stage: MockStage;
}

/** One step of a distillation workflow: its status, progress, key numbers and the job behind it. */
export const DistillationStageCard: FC<DistillationStageCardProps> = ({ stage }) => (
  <Panel
    slotHeading={
      <Flex align="center" gap="density-md">
        <Text kind="title/xs">{stage.title}</Text>
        <StatusBadge status={stage.status} />
        {stage.status === 'active' && (
          <Spinner size="small" aria-label={`${stage.title} running`} />
        )}
      </Flex>
    }
    elevation="high"
    density="compact"
    className="w-full"
    data-testid={`distillation-stage-${stage.id}`}
  >
    <Stack gap="density-lg">
      <Text kind="body/regular/sm" className="text-secondary">
        {stage.description}
      </Text>
      {stage.status !== 'pending' && (
        <ProgressBar aria-label={`${stage.title} progress`} value={stage.progress} />
      )}
      <Flex gap="density-3xl" wrap="wrap">
        {stage.metrics.map((metric) => (
          <Stack key={metric.label} gap="density-xs">
            <Text kind="body/regular/xs" className="text-secondary">
              {metric.label}
            </Text>
            <Text kind="body/semibold/md">{metric.value}</Text>
          </Stack>
        ))}
        {stage.jobRef && (
          <Stack gap="density-xs">
            <Text kind="body/regular/xs" className="text-secondary">
              {stage.jobRef.kind}
            </Text>
            <Text kind="mono/sm">{stage.jobRef.name}</Text>
          </Stack>
        )}
      </Flex>
    </Stack>
  </Panel>
);
