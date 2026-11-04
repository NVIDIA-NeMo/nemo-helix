// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { EvalJobDeployment } from '@studio/api/evaluation/utils';
import type { FC } from 'react';

const hostOf = (url: string): string => {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
};

interface EvalDeploymentValueProps {
  deployment: EvalJobDeployment | null | undefined;
}

export const EvalDeploymentValue: FC<EvalDeploymentValueProps> = ({ deployment }) => {
  if (!deployment) {
    return <Text>—</Text>;
  }
  if (deployment.kind === 'custom') {
    return <Text title={deployment.url}>{hostOf(deployment.url)}</Text>;
  }
  if (deployment.kind === 'default') {
    return (
      <Text title="This run called the agent's default route, so each request went to whichever deployment was running">
        Default
      </Text>
    );
  }
  return <Text>{deployment.name}</Text>;
};
