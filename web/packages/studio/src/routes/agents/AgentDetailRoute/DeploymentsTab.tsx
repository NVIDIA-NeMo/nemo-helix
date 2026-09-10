// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Anchor, Button, Flex, Stack, StatusIndicator, Text } from '@nvidia/foundations-react-core';
import { type AgentSpecSource, githubCommitUrl } from '@studio/api/agents/useAgentSpecFileset';
import {
  deploymentStatusColor,
  shortRevision,
} from '@studio/routes/agents/AgentDetailRoute/helpers';
import { NoHealthyDeploymentsBanner } from '@studio/routes/agents/AgentDetailRoute/NoHealthyDeploymentsBanner';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import type { FC } from 'react';

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
}

/** A commit, linked to GitHub when the source it came from is still known. */
const CommitLink: FC<{ source?: AgentSpecSource; revision: string }> = ({ source, revision }) =>
  source ? (
    <Anchor
      href={githubCommitUrl(source.owner, source.repo, revision)}
      target="_blank"
      rel="noreferrer noopener"
      textKind="body/regular/xs"
      underline
      className="text-brand"
    >
      {shortRevision(revision)}
    </Anchor>
  ) : (
    <>{shortRevision(revision)}</>
  );

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
}) => (
  <Stack gap="5" className="w-full">
    <DetailPanel title="Deployments" flush>
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
            <Flex
              key={deployment.name}
              align="center"
              gap="2"
              className={`px-4 py-3 ${index > 0 ? 'border-t border-base' : ''}`}
            >
              <StatusIndicator color={deploymentStatusColor(deployment.status)} size="small" />
              <Stack gap="0" className="min-w-0 flex-1">
                <Text kind="body/semibold/sm">{deployment.name}</Text>
                {deployment.endpoint && (
                  <Text kind="body/regular/xs" color="secondary" className="truncate">
                    {deployment.endpoint}
                  </Text>
                )}
                {deployment.error && (
                  <Text kind="body/regular/xs" color="danger" className="truncate">
                    {deployment.error}
                  </Text>
                )}
                {deployment.spec_revision ? (
                  <Text kind="body/regular/xs" color="secondary" className="truncate">
                    Staged from commit{' '}
                    <CommitLink source={specSource} revision={deployment.spec_revision} />
                    {specSource && deployment.spec_revision !== specSource.revision ? (
                      <>
                        {' · source is now '}
                        <CommitLink source={specSource} revision={specSource.revision} />
                      </>
                    ) : null}
                  </Text>
                ) : null}
              </Stack>
              <StatusBadge status={deployment.status} />
              <Flex gap="1" className="shrink-0">
                <Button
                  kind="tertiary"
                  size="small"
                  disabled={deployment.status !== 'running'}
                  onClick={() => onChat(deployment)}
                >
                  Chat
                </Button>
                <Button kind="tertiary" size="small" onClick={() => onViewLogs(deployment)}>
                  Logs
                </Button>
                <Button
                  kind="tertiary"
                  size="small"
                  color="danger"
                  onClick={() => onDelete(deployment)}
                >
                  Delete
                </Button>
              </Flex>
            </Flex>
          ))}
        </Stack>
      )}
    </DetailPanel>
  </Stack>
);
