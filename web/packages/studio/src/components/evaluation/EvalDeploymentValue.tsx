// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { EvalJobDeployment } from '@studio/api/evaluation/utils';
import { getAgentDeploymentDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

interface EvalDeploymentValueProps {
  workspace: string;
  deployment: EvalJobDeployment | null;
}

export const EvalDeploymentValue: FC<EvalDeploymentValueProps> = ({ workspace, deployment }) => {
  if (!deployment) return <>-</>;
  if (deployment.kind === 'automatic') {
    return (
      <Text kind="body/regular/sm" title="The gateway picked a running deployment for each request">
        Automatic
      </Text>
    );
  }
  return (
    <Link
      to={getAgentDeploymentDetailRoute(workspace, deployment.name)}
      className="text-primary underline"
    >
      {deployment.name}
    </Link>
  );
};
