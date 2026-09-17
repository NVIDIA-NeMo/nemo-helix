// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Stack } from '@nvidia/foundations-react-core';
import type { AgentSpecSource } from '@studio/api/agents/useAgentSpecFileset';
import { AGENT_CONTAINER_DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import { DeploymentRow } from '@studio/routes/agents/AgentDetailRoute/components/DeploymentRow';
import { NoHealthyDeploymentsBanner } from '@studio/routes/agents/AgentDetailRoute/NoHealthyDeploymentsBanner';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { PackageAgentControl } from '@studio/routes/agents/AgentDetailRoute/PackageAgentControl';
import { type FC } from 'react';

interface DeploymentsTabProps {
  agentName?: string;
  deployments: AgentDeployment[];
  isDeploymentsLoading: boolean;
  isDeploying: boolean;
  onDeploy: () => void;
  onChat: (deployment: AgentDeployment) => void;
  onDelete: (deployment: AgentDeployment) => void;
  onViewLogs: (deployment: AgentDeployment) => void;
  /** Deploying requires a Platform-managed agent config (Fabric integration). */
  canDeploy: boolean;
  /** Where the agent's files come from, to link each staged commit and mark stale ones. */
  specSource?: AgentSpecSource;
  workspace: string;
  /** Packaging is Fabric-only, a narrower gate than `canDeploy`. */
  canPackage: boolean;
  /** Still resolving whether this agent can be packaged, which is not the same as "no". */
  isAgentLoading?: boolean;
  onImageBuilt?: (image: string) => void;
  onImageAvailable?: (image: string) => void;
}

/** Deployments list with per-deployment actions. */
export const DeploymentsTab: FC<DeploymentsTabProps> = ({
  agentName,
  deployments,
  isDeploymentsLoading,
  isDeploying,
  onDeploy,
  onChat,
  onDelete,
  onViewLogs,
  canDeploy,
  specSource,
  workspace,
  canPackage,
  isAgentLoading,
  onImageBuilt,
  onImageAvailable,
}) => (
  <Stack gap="5" className="w-full">
    <DetailPanel
      title="Deployments"
      flush
      slotAction={
        agentName && AGENT_CONTAINER_DEPLOYMENTS_ENABLED ? (
          <PackageAgentControl
            // The route is reused across agents; without this the control would
            // report the previous agent's build.
            key={agentName}
            workspace={workspace}
            agentName={agentName}
            canPackage={canPackage}
            isAgentLoading={isAgentLoading}
            onImageBuilt={onImageBuilt}
            onImageAvailable={onImageAvailable}
          />
        ) : null
      }
    >
      {!isDeploymentsLoading && deployments.length === 0 ? (
        <div className="p-4">
          <NoHealthyDeploymentsBanner
            agentName={agentName}
            isDeploying={isDeploying}
            onDeploy={onDeploy}
            canDeploy={canDeploy}
            message="No deployments for this agent."
          />
        </div>
      ) : (
        <Stack gap="0">
          {deployments.map((deployment, index) => (
            <DeploymentRow
              key={deployment.name}
              deployment={deployment}
              isFirst={index === 0}
              specSource={specSource}
              onChat={onChat}
              onDelete={onDelete}
              onViewLogs={onViewLogs}
            />
          ))}
        </Stack>
      )}
    </DetailPanel>
  </Stack>
);
