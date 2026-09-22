// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Button, Flex, Text } from '@nvidia/foundations-react-core';
import { Loader2 } from 'lucide-react';
import type { FC } from 'react';

interface NoHealthyDeploymentsBannerProps {
  agentName?: string;
  isDeploying: boolean;
  onDeploy: () => void;
  message?: string;
  canDeploy: boolean;
}

export const NoHealthyDeploymentsBanner: FC<NoHealthyDeploymentsBannerProps> = ({
  agentName,
  isDeploying,
  onDeploy,
  canDeploy,
  message = 'No healthy deployments available to chat with.',
}) => (
  <Banner
    kind="inline"
    status="warning"
    slotActions={
      isDeploying ? (
        <Flex align="center" className="h-full" gap="2">
          <Loader2 size={14} className="animate-spin" aria-label="Deploying agent" />
          <Text kind="label/regular/sm">Deploying…</Text>
        </Flex>
      ) : (
        <Button
          kind="secondary"
          size="small"
          disabled={!agentName || !canDeploy}
          onClick={onDeploy}
        >
          Deploy this Agent
        </Button>
      )
    }
  >
    {canDeploy
      ? message
      : `${message} Integrate this agent with NeMo Helix to enable deployment.`}
  </Banner>
);
