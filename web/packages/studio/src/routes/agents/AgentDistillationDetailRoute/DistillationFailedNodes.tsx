// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { RuntimeNode } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { FC } from 'react';

export interface DistillationFailedNodesProps {
  nodes: Record<string, RuntimeNode>;
  onRetry: (node: string) => void;
  isRetrying: boolean;
}

/** The workflow nodes that failed, each with its error and a retry. Renders nothing when none did. */
export const DistillationFailedNodes: FC<DistillationFailedNodesProps> = ({
  nodes,
  onRetry,
  isRetrying,
}) => {
  const failed = Object.entries(nodes).filter(([, node]) => node.status === 'failed');
  if (failed.length === 0) return null;

  return (
    <Panel slotHeading="Failed steps" elevation="high" density="compact" className="w-full">
      <Stack gap="density-lg">
        {failed.map(([key, node]) => (
          <Flex key={key} justify="between" align="center" gap="density-lg" wrap="wrap">
            <Stack gap="density-xs">
              <Text kind="mono/sm">{key}</Text>
              <Text kind="body/regular/sm" className="text-feedback-danger">
                {node.error ?? 'No error message'}
              </Text>
            </Stack>
            <Button
              kind="secondary"
              size="small"
              disabled={isRetrying}
              onClick={() => onRetry(key)}
            >
              Retry
            </Button>
          </Flex>
        ))}
      </Stack>
    </Panel>
  );
};
