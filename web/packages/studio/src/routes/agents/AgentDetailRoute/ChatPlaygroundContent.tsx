// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { handleGenericError } from '@nemo/common/src/utils/logger';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Block, Select } from '@nvidia/foundations-react-core';
import { useAgentChatSession } from '@studio/api/agents/useAgentChatSession';
import { ModelChat } from '@studio/components/ModelChat';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { NoHealthyDeploymentsBanner } from '@studio/routes/agents/AgentDetailRoute/NoHealthyDeploymentsBanner';
import type { FC, RefObject } from 'react';

interface ChatPlaygroundContentProps {
  workspace: string;
  agentName?: string;
  chatDeployment?: AgentDeployment;
  healthyDeployments: AgentDeployment[];
  isDeploymentsLoading: boolean;
  isDeploying: boolean;
  chatAreaRef: RefObject<HTMLDivElement | null>;
  onSelectDeployment: (name: string) => void;
  onDeploy: () => void;
  /** Deploying requires a Platform-managed agent config (Fabric integration). */
  canDeploy: boolean;
}

export const ChatPlaygroundContent: FC<ChatPlaygroundContentProps> = ({
  workspace,
  agentName,
  chatDeployment,
  healthyDeployments,
  isDeploymentsLoading,
  isDeploying,
  chatAreaRef,
  onSelectDeployment,
  onDeploy,
  canDeploy,
}) => {
  const deploymentSelectItems = healthyDeployments.flatMap((d) =>
    d.name
      ? [
          {
            value: d.name,
            children: d.status ? `${d.name} · ${d.status}` : d.name,
          },
        ]
      : []
  );
  const noHealthyDeployments = !isDeploymentsLoading && healthyDeployments.length === 0;
  const {
    extraHeaders,
    isPending: isSessionPending,
    recoverFromError,
  } = useAgentChatSession(workspace, chatDeployment);

  const handleChatError = (error: Error) => {
    if (!recoverFromError(error)) handleGenericError(error);
  };

  return (
    <div ref={chatAreaRef} className="flex flex-col h-full min-h-0">
      {!noHealthyDeployments && healthyDeployments.length > 1 && (
        <Block padding="4" className="border-b border-base shrink-0">
          <Select
            value={chatDeployment?.name ?? ''}
            items={deploymentSelectItems}
            onValueChange={(v) => onSelectDeployment(v)}
          />
        </Block>
      )}
      {noHealthyDeployments && (
        <Block padding="4" className="shrink-0">
          <NoHealthyDeploymentsBanner
            agentName={agentName}
            isDeploying={isDeploying}
            onDeploy={onDeploy}
            canDeploy={canDeploy}
          />
        </Block>
      )}
      <Block className="flex-1 min-h-0" padding="4">
        <ModelChat
          model={chatDeployment?.name ?? agentName ?? ''}
          workspace={workspace}
          baseURL={
            chatDeployment
              ? `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/${workspace}/deployments/${chatDeployment.name}/-/v1`
              : undefined
          }
          extraHeaders={extraHeaders}
          onError={handleChatError}
          disabled={
            isDeploymentsLoading || noHealthyDeployments || !chatDeployment || isSessionPending
          }
        />
      </Block>
    </div>
  );
};
