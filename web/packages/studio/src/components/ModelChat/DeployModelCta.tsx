// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { ResourceRef } from '@nemo/common/src/types';
import { Button } from '@nvidia/foundations-react-core';
import { DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import { getWorkspaceNewDeploymentRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

export interface DeployModelCtaProps {
  /** `<workspace>/<name>` reference to the model entity to deploy. */
  modelRef: ResourceRef;
  /**
   * Button copy. Override when `modelRef` is not the entity the user is looking
   * at — e.g. an adapter's chat, where the deployable entity is its base model.
   */
  label?: string;
}

/**
 * Sends the user to the Create Deployment wizard with the Workspace source
 * pre-selected for `modelRef`.
 *
 * Renders nothing when deployments are disabled — the wizard route is gated on
 * the same flag, so the button would lead nowhere.
 */
export const DeployModelCta: FC<DeployModelCtaProps> = ({
  modelRef,
  label = 'Deploy this model',
}) => {
  const navigate = useNavigate();

  if (!DEPLOYMENTS_ENABLED) return null;

  const { workspace } = getPartsFromReference(modelRef);

  return (
    <Button
      color="brand"
      onClick={() => navigate(getWorkspaceNewDeploymentRoute(workspace, { model: modelRef }))}
    >
      {label}
    </Button>
  );
};
