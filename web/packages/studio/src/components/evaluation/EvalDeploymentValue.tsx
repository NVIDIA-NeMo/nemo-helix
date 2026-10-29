// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { EvalJobDeployment } from '@studio/api/evaluation/utils';
import { getAgentDeploymentDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

interface EvalDeploymentValueProps {
  deployment: EvalJobDeployment | null | undefined;
}

export const EvalDeploymentValue: FC<EvalDeploymentValueProps> = ({ deployment }) => {
  if (!deployment) {
    return <Text>—</Text>;
  }
  if (deployment.kind === 'automatic') {
    return <Text title="The gateway picked a running deployment for each request">Automatic</Text>;
  }
  return (
    <Link
      to={getAgentDeploymentDetailRoute(deployment.workspace, deployment.name)}
      className="text-primary underline"
    >
      {deployment.name}
    </Link>
  );
};
