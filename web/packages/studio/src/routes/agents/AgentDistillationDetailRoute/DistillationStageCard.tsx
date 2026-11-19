// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { Flex, Panel, ProgressBar, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import type { StageNode } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationNodes';
import type { MockStage } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import { DistillationNodeRow } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationNodeRow';
import type { FC } from 'react';

export interface DistillationStageCardProps {
  workspace: string;
  stage: MockStage;
  nodes: StageNode[];
  onRetry: (node: string) => void;
  isRetrying: boolean;
}

/** One step of a distillation workflow: its status, progress, key numbers and the job behind it. */
export const DistillationStageCard: FC<DistillationStageCardProps> = ({
  workspace,
  stage,
  nodes,
  onRetry,
  isRetrying,
}) => (
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
      </Flex>
      {nodes.length > 0 && (
        <Stack gap="density-lg">
          {nodes.map((node) => (
            <DistillationNodeRow
              key={node.key}
              workspace={workspace}
              node={node}
              onRetry={onRetry}
              isRetrying={isRetrying}
            />
          ))}
        </Stack>
      )}
    </Stack>
  </Panel>
);
