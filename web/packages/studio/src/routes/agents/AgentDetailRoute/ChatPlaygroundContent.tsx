// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describeChatError } from '@nemo/common/src/components/AssistantChat/describeChatError';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Banner, Block } from '@nvidia/foundations-react-core';
import { useMcpStatus } from '@studio/api/agents/useMcpStatus';
import { ModelChat } from '@studio/components/ModelChat';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { NoHealthyDeploymentsBanner } from '@studio/routes/agents/AgentDetailRoute/NoHealthyDeploymentsBanner';
import type { FC, RefObject } from 'react';

interface ChatPlaygroundContentProps {
  workspace: string;
  agentName?: string;
  /** The header's selection, any status; chat is only enabled when it is running. */
  deployment?: AgentDeployment;
  hasHealthyDeployments: boolean;
  isDeploymentsLoading: boolean;
  isDeploying: boolean;
  chatAreaRef: RefObject<HTMLDivElement | null>;
  onDeploy: () => void;
  /** Deploying requires a Platform-managed agent config (Fabric integration). */
  canDeploy: boolean;
  isAgentPending?: boolean;
}

export const ChatPlaygroundContent: FC<ChatPlaygroundContentProps> = ({
  workspace,
  agentName,
  deployment,
  hasHealthyDeployments,
  isDeploymentsLoading,
  isDeploying,
  chatAreaRef,
  onDeploy,
  canDeploy,
  isAgentPending,
}) => {
  const chatDeployment = deployment?.status === 'running' ? deployment : undefined;
  const showBanner = !isDeploymentsLoading && !chatDeployment;
  const { data: mcpStatus } = useMcpStatus(workspace, chatDeployment?.name);
  const downServers = mcpStatus?.servers.filter(
    ({ state }) => state === 'failed' || state === 'unresolved'
  );

  return (
    <div ref={chatAreaRef} className="mx-auto flex h-full min-h-0 w-full max-w-4xl flex-col">
      {showBanner && (
        <Block padding="4" className="shrink-0">
          <NoHealthyDeploymentsBanner
            agentName={agentName}
            isDeploying={isDeploying}
            onDeploy={onDeploy}
            canDeploy={canDeploy}
            isAgentPending={isAgentPending}
            message={
              deployment && hasHealthyDeployments
                ? `${deployment.name} is ${deployment.status ?? 'unavailable'}. Switch to a running deployment to chat.`
                : undefined
            }
          />
        </Block>
      )}
      {downServers?.length ? (
        <Block padding="4" className="shrink-0 pb-0">
          <Banner kind="inline" status="error" className="[overflow-wrap:anywhere]">
            {`MCP ${downServers.length === 1 ? 'server' : 'servers'} ${downServers.map(({ name }) => name).join(', ')} failed to start. The agent runs without ${downServers.length === 1 ? 'its' : 'their'} tools.`}
          </Banner>
        </Block>
      ) : null}
      <Block className="flex-1 min-h-0" padding="4">
        <ModelChat
          model={chatDeployment?.name ?? agentName ?? ''}
          workspace={workspace}
          baseURL={
            chatDeployment
              ? `${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/${workspace}/deployments/${chatDeployment.name}/-/v1`
              : undefined
          }
          disabled={isDeploymentsLoading || !chatDeployment}
          describeError={describeChatError}
        />
      </Block>
    </div>
  );
};
