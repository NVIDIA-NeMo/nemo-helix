// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ExpandableMessage } from '@nemo/common/src/components/ExpandableMessage';
import { ExternalLink } from '@nemo/common/src/components/ExternalLink';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { StatusDot } from '@nemo/common/src/components/StatusDot';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useAgentsGetDeployment } from '@nemo/sdk/generated/agents/agent-deployments';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import { Badge, Block, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { type AgentSpecSource, githubCommitUrl } from '@studio/api/agents/useAgentSpecFileset';
import {
  deploymentModeLabel,
  deploymentUrl,
  shortRevision,
} from '@studio/routes/agents/AgentDetailRoute/helpers';
import { type FC } from 'react';

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
 * One deployment. A failure or deletion message is the row's most useful content
 * and is often longer than the row, so it truncates with a "Show more" toggle.
 *
 * The list endpoint withholds `error` (it can carry process logs), so failed and
 * deleting rows read it from the detail endpoint, which requires read access.
 */
export const DeploymentRow: FC<DeploymentRowProps> = ({
  deployment,
  isFirst,
  specSource,
  onChat,
  onDelete,
  onViewLogs,
}) => {
  const url = deploymentUrl(deployment);
  const isDeleting = deployment.status === 'deleting';
  const { workspace = '', name = '' } = deployment;
  const { data: detail } = useAgentsGetDeployment(workspace, name, {
    query: {
      enabled: !!workspace && !!name && (isDeleting || deployment.status === 'failed'),
      refetchInterval: isDeleting ? JOB_POLLING_INTERVAL_MS : false,
    },
  });
  const message = detail?.error || deployment.error;

  return (
    <Flex align="start" gap="2" className={`px-4 py-3 ${isFirst ? '' : 'border-t border-base'}`}>
      <StatusDot status={deployment.status} className="mt-1.5" />
      <Stack gap="0" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm">{deployment.name}</Text>
        {url && (
          <Text kind="body/regular/xs" className="truncate text-secondary">
            {url}
          </Text>
        )}
        {/* An agent has many images over its life; without this the row gives no way
            to tell which one is actually running. */}
        {deployment.image && (
          <Text
            kind="body/regular/xs"
            className="truncate font-mono text-secondary"
            title={deployment.image}
          >
            {deployment.image}
          </Text>
        )}
        {message && (
          <Block className="mt-density-xs">
            <ExpandableMessage
              errorMessage={message}
              characterLimit={150}
              attributes={{
                Text: {
                  kind: 'body/regular/xs',
                  className: 'text-feedback-danger whitespace-pre-wrap break-words',
                },
              }}
            />
          </Block>
        )}
        {deployment.spec_revision ? (
          <Text kind="body/regular/xs" className="truncate text-secondary">
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
        <Badge kind="outline" color="gray" size="small">
          {deploymentModeLabel(deployment.deployment_mode)}
        </Badge>
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
