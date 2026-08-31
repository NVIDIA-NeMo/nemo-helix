// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { LogViewer } from '@nemo/common/src/components/LogViewer';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { usePackageAgent } from '@studio/api/agents/usePackageAgent';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import type { FC } from 'react';

interface PackageAgentPanelProps {
  workspace: string;
  agentName: string;
  /** Packaging needs a Platform-managed agent config, the same gate deploying uses. */
  canPackage: boolean;
  /** Offers the finished tag to the deployment flow. */
  onImageBuilt?: (image: string) => void;
}

/**
 * Build a container image for this agent so a docker or k8s deployment has one.
 *
 * The build runs for minutes, so this stays on the page rather than inside the
 * deployment modal, and streams the job's logs while it runs.
 */
export const PackageAgentPanel: FC<PackageAgentPanelProps> = ({
  workspace,
  agentName,
  canPackage,
  onImageBuilt,
}) => {
  const {
    packageAgent,
    submitError,
    isSubmitting,
    jobName,
    isRunning,
    isComplete,
    isFailed,
    logs,
    isLogsLoading,
    image,
    published,
  } = usePackageAgent({ workspace, agentName });

  return (
    <DetailPanel
      title="Container image"
      slotAction={
        <Button
          kind="secondary"
          size="small"
          disabled={!canPackage || isSubmitting || isRunning}
          onClick={() => packageAgent({})}
        >
          {isRunning ? 'Building…' : 'Build image'}
        </Button>
      }
    >
      <Stack gap="3">
        {!canPackage ? (
          <Text kind="body/regular/sm" className="text-secondary">
            Packaging is available for Platform-managed agents. Build a NAT workflow image with{' '}
            <code>nemo agents package</code>.
          </Text>
        ) : null}

        {submitError ? (
          <Text kind="body/regular/sm" className="text-danger">
            {getErrorMessage(submitError as Error, 'Failed to start the packaging job')}
          </Text>
        ) : null}

        {!jobName && canPackage && !submitError ? (
          <Text kind="body/regular/sm" className="text-secondary">
            Build an image for this agent to deploy it with Docker or Kubernetes.
          </Text>
        ) : null}

        {isComplete && image ? (
          <Stack gap="1">
            <Text kind="body/bold/sm">{image}</Text>
            {published ? (
              <Text kind="body/regular/sm" className="text-secondary">
                Pushed to {published}
              </Text>
            ) : null}
            {onImageBuilt ? (
              <Flex>
                <Button kind="tertiary" size="small" onClick={() => onImageBuilt(image)}>
                  Use for deployment
                </Button>
              </Flex>
            ) : null}
          </Stack>
        ) : null}

        {isComplete && !image ? (
          <Text kind="body/regular/sm" className="text-secondary">
            The job finished without reporting an image tag. Check the logs below.
          </Text>
        ) : null}

        {isFailed ? (
          <Text kind="body/regular/sm" className="text-danger">
            Packaging failed. The build output is below.
          </Text>
        ) : null}

        {jobName ? (
          <LogViewer
            logs={logs}
            isLoading={isLogsLoading}
            downloadFilename={`${agentName}-package.log`}
            emptyMessage="Waiting for build output…"
          />
        ) : null}
      </Stack>
    </DetailPanel>
  );
};
