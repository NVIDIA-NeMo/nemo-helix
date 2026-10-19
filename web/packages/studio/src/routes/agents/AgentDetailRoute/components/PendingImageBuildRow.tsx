// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { JOBS_ENABLED } from '@studio/constants/environment';
import type { PendingImageBuild } from '@studio/routes/agents/AgentDetailRoute/BuildThenDeploy';
import { deploymentModeLabel } from '@studio/routes/agents/AgentDetailRoute/helpers';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { TriangleAlert } from 'lucide-react';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

interface PendingImageBuildRowProps {
  workspace: string;
  build: PendingImageBuild;
}

/** Stands in for the deployment a build-then-deploy request creates once its image is ready. */
export const PendingImageBuildRow: FC<PendingImageBuildRowProps> = ({ workspace, build }) => {
  const navigate = useNavigate();
  const { jobName } = build;
  return (
    <Flex align="start" gap="2" className="px-4 py-3" data-testid="pending-image-build">
      {build.isStalled ? (
        <TriangleAlert
          className="mt-0.5 size-4 shrink-0 text-feedback-warning"
          aria-label="Image build not started"
        />
      ) : (
        <Spinner size="small" aria-label="Building image" className="mt-0.5 shrink-0" />
      )}
      <Stack gap="0" className="min-w-0 flex-1">
        <Text kind="body/semibold/sm">
          {build.isStalled ? 'Waiting for the image build to start' : 'Building an image to deploy'}
        </Text>
        {build.isStalled ? (
          <Text kind="body/regular/xs" className="text-feedback-warning">
            The build was accepted but has not started. Check that the platform is running a jobs
            controller.
          </Text>
        ) : (
          <Text kind="body/regular/xs" className="text-secondary">
            Deploys when the build finishes, usually in a few minutes. Keep this page open until
            then.
          </Text>
        )}
      </Stack>
      <Flex align="center" gap="2" className="shrink-0">
        <Badge kind="outline" color="gray" size="small">
          {deploymentModeLabel(build.mode)}
        </Badge>
        {jobName && JOBS_ENABLED ? (
          <Button
            kind="tertiary"
            size="small"
            onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, jobName))}
          >
            View job
          </Button>
        ) : null}
      </Flex>
    </Flex>
  );
};
