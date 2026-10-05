// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import type { DeploymentMode } from '@studio/api/agents/useDeploymentModes';
import { deploymentModeLabel } from '@studio/routes/agents/AgentDetailRoute/helpers';
import type { FC } from 'react';

interface DeploymentModeSelectProps extends UseControllerComponentProps {
  modes: readonly DeploymentMode[];
  loading?: boolean;
}

export const DeploymentModeSelect: FC<DeploymentModeSelectProps> = ({
  modes,
  loading,
  useControllerProps,
  formFieldProps,
}) => (
  <ControlledSelect
    useControllerProps={useControllerProps}
    loading={loading}
    items={modes.map((mode) => ({ value: mode, children: deploymentModeLabel(mode) }))}
    formFieldProps={{ slotLabel: 'Runtime', ...formFieldProps }}
  />
);
