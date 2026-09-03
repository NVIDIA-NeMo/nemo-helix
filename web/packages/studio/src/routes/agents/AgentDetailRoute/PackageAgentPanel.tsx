// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { usePackageAgent } from '@studio/api/agents/usePackageAgent';
import { JOBS_ENABLED } from '@studio/constants/environment';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { useEffect, type FC } from 'react';
import { useNavigate } from 'react-router';

interface PackageAgentPanelProps {
  workspace: string;
  agentName: string;
  /** Packaging needs a Platform-managed agent config, the same gate deploying uses. */
  canPackage: boolean;
  /** Offers the finished tag to the deployment flow. */
  onImageBuilt?: (image: string) => void;
  /**
   * Reports the finished tag as soon as the build produces one, so deploying
   * from anywhere on the page starts from the image this agent just built.
   */
  onImageAvailable?: (image: string) => void;
}

/**
 * Build a container image for this agent so a docker or k8s deployment has one.
 *
 * The build runs for minutes, so this stays on the page rather than inside the
 * deployment modal. Build output belongs on the job page, which already renders
 * it in full, rather than filling this panel with a wall of install lines.
 */
export const PackageAgentPanel: FC<PackageAgentPanelProps> = ({
  workspace,
  agentName,
  canPackage,
  onImageBuilt,
  onImageAvailable,
}) => {
  const navigate = useNavigate();
  const {
    packageAgent,
    submitError,
    isSubmitting,
    jobName,
    isRunning,
    isQueued,
    isStalled,
    isComplete,
    isFailed,
    isRestored,
    image,
    published,
  } = usePackageAgent({ workspace, agentName });

  useEffect(() => {
    if (isComplete && image) {
      onImageAvailable?.(image);
    }
  }, [isComplete, image, onImageAvailable]);

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
          {isQueued ? 'Queued…' : isRunning ? 'Building…' : 'Build image'}
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
            <Text kind="body/bold/sm" className="text-success">
              {isRestored ? 'Image ready' : 'Build finished — image ready'}
            </Text>
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
            The job finished without reporting an image tag. Open the job to see why.
          </Text>
        ) : null}

        {isStalled ? (
          <Text kind="body/regular/sm" className="text-warning">
            The job was accepted but has not started. Check that the platform is running a jobs
            controller.
          </Text>
        ) : null}

        {isFailed ? (
          <Text kind="body/regular/sm" className="text-danger">
            Packaging failed. Open the job for the build output.
          </Text>
        ) : null}

        {(isQueued || isRunning) && !isStalled ? (
          <Flex gap="2" className="items-center">
            <Spinner size="small" aria-label="Building image" />
            <Text kind="body/regular/sm" className="text-secondary">
              {isQueued ? 'Waiting for a build to start…' : 'Building — this takes a few minutes.'}
            </Text>
          </Flex>
        ) : null}

        {jobName && JOBS_ENABLED ? (
          <Flex>
            <Button
              kind="tertiary"
              size="small"
              onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, jobName))}
            >
              View job
            </Button>
          </Flex>
        ) : null}
      </Stack>
    </DetailPanel>
  );
};
