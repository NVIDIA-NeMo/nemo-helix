// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner } from '@nvidia/foundations-react-core';
import type { DeploymentMode } from '@studio/api/agents/useDeploymentModes';
import { deploymentModeLabel } from '@studio/routes/agents/AgentDetailRoute/helpers';
import type { FC } from 'react';

export const ImageBuildFirstNotice: FC<{ mode: DeploymentMode }> = ({ mode }) => (
  <Banner kind="inline" status="info">
    {deploymentModeLabel(mode)} runs a container image, and none is set. Studio builds one for this
    agent, then deploys it when the build finishes, usually in a few minutes. Keep the agent&apos;s
    page open until then.
  </Banner>
);
