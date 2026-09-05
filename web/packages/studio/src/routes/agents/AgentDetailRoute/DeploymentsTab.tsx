// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExternalLink } from '@nemo/common/src/components/ExternalLink';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Button, Flex, Stack, StatusIndicator, Text } from '@nvidia/foundations-react-core';
import { type AgentSpecSource, githubCommitUrl } from '@studio/api/agents/useAgentSpecFileset';
import { AGENT_CONTAINER_DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import {
  deploymentStatusColor,
  shortRevision,
} from '@studio/routes/agents/AgentDetailRoute/helpers';
import { NoHealthyDeploymentsBanner } from '@studio/routes/agents/AgentDetailRoute/NoHealthyDeploymentsBanner';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { PackageAgentControl } from '@studio/routes/agents/AgentDetailRoute/PackageAgentControl';
import { useState, type FC } from 'react';

interface DeploymentRowProps {
  deployment: AgentDeployment;
  isFirst: boolean;
  /** Where the agent's files come from, to link the staged commit and mark it stale. */
  specSource?: AgentSpecSource;
  onChat: (deployment: AgentDeployment) => void;
  onDelete: (deployment: AgentDeployment) => void;
  onViewLogs: (deployment: AgentDeployment) => void;
}

/** A commit, linked to GitHub when the source it came from is still known. */
const CommitLink: FC<{ source?: AgentSpecSource; revision: string }> = ({ source, revision }) =>
  source ? (
    <ExternalLink
      href={githubCommitUrl(source.owner, source.repo, revision)}
      textKind="body/regular/xs"
    >
      {shortRevision(revision)}
    </ExternalLink>
  ) : (
    <>{shortRevision(revision)}</>
  );

/**
 * One deployment. A failure message is the row's most useful content and is
 * often longer than the row, so it collapses to two lines with a toggle rather
 * than being ellipsised into uselessness.
 */
const DeploymentRow: FC<DeploymentRowProps> = ({
  deployment,
  isFirst,
  specSource,
  onChat,
  onDelete,
  onViewLogs,
}) => {
  const [isErrorExpanded, setIsErrorExpanded] = useState(false);

  return (
    <Flex align="start" gap="2" className={`px-4 py-3 ${isFirst ? '' : 'border-t border-base'}`}>
      <StatusIndicator
        color={deploymentStatusColor(deployment.status)}
        size="small"
        className="mt-1.5 shrink-0"
      />
      <Stack gap="0" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm">{deployment.name}</Text>
        {deployment.endpoint && (
          <Text kind="body/regular/xs" color="secondary" className="truncate">
            {deployment.endpoint}
          </Text>
        )}
        {/* An agent has many images over its life; without this the row gives no way
            to tell which one is actually running. */}
        {deployment.image && (
          <Text
            kind="body/regular/xs"
            color="secondary"
            className="truncate font-mono"
            title={deployment.image}
          >
            {deployment.image}
          </Text>
        )}
        {deployment.error && (
          <Stack gap="density-xs" className="mt-density-xs items-start">
            <Text
              kind="body/regular/xs"
              color="danger"
              className={isErrorExpanded ? 'whitespace-pre-wrap break-words' : 'line-clamp-2'}
            >
              {deployment.error}
            </Text>
            <Button
              kind="tertiary"
              size="tiny"
              aria-expanded={isErrorExpanded}
              onClick={() => setIsErrorExpanded((open) => !open)}
            >
              {isErrorExpanded ? 'Show less' : 'Show full error'}
            </Button>
          </Stack>
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
      <Flex align="center" gap="2" className="shrink-0">
        <StatusBadge status={deployment.status} />
        <Flex gap="1">
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
          <Button kind="tertiary" size="small" color="danger" onClick={() => onDelete(deployment)}>
            Delete
          </Button>
        </Flex>
      </Flex>
    </Flex>
  );
};

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
