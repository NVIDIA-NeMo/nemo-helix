// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { Button, Flex, ProgressBar, Stack, Text } from '@nvidia/foundations-react-core';
import type { StageNode } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationNodes';
import { useTrainingProgress } from '@studio/routes/agents/AgentDetailRoute/distillation/useTrainingProgress';
import type { FC } from 'react';

export interface DistillationNodeRowProps {
  workspace: string;
  node: StageNode;
  onRetry: (node: string) => void;
  isRetrying: boolean;
}

/** One job of a stage: its status, the platform job behind it, live training progress, and a retry on failure. */
export const DistillationNodeRow: FC<DistillationNodeRowProps> = ({
  workspace,
  node,
  onRetry,
  isRetrying,
}) => {
  const training = useTrainingProgress(
    workspace,
    node.job,
    node.status === 'running',
    node.kind === 'customization_job' && node.status !== 'pending'
  );

  return (
    <Stack gap="density-sm" data-testid={`distillation-node-${node.key}`}>
      <Flex justify="between" align="center" gap="density-lg" wrap="wrap">
        <Flex align="center" gap="density-md" wrap="wrap">
          <Text kind="mono/sm">{node.key}</Text>
          <StatusBadge status={node.status} />
          {node.attempt > 1 && (
            <Text kind="body/regular/xs" className="text-secondary">
              attempt {node.attempt}
            </Text>
          )}
          {node.job && (
            <Text kind="mono/sm" className="text-secondary">
              {node.job}
            </Text>
          )}
        </Flex>
        {node.status === 'failed' && (
          <Button kind="secondary" size="small" disabled={isRetrying} onClick={() => onRetry(node.key)}>
            Retry
          </Button>
        )}
      </Flex>
      {training?.percent !== undefined && (
        <ProgressBar aria-label={`${node.key} training progress`} value={training.percent} />
      )}
      {training?.detail && (
        <Text kind="body/regular/xs" className="text-secondary">
          {training.detail}
        </Text>
      )}
      {node.error && (
        <Text kind="body/regular/sm" className="text-feedback-danger">
          {node.error}
        </Text>
      )}
    </Stack>
  );
};
