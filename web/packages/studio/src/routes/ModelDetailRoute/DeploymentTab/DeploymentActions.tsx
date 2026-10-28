// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import type { ResourceRef } from '@nemo/common/src/types';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex } from '@nvidia/foundations-react-core';
import { DeployModelCta } from '@studio/components/ModelChat/DeployModelCta';
import { LIVE_STATUSES } from '@studio/hooks/useBaseModelDeploymentReadiness';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { ModelDetailTab } from '@studio/routes/ModelDetailRoute/constants';
import { DeleteDeploymentButton } from '@studio/routes/ModelDetailRoute/DeploymentTab/DeleteDeploymentButton';
import { getModelDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

export interface DeploymentActionsProps {
  model: ModelEntity;
  adapter: Adapter | null;
  /** The base model's deployment state, as the Deployment tab shows it. */
  deployment: DeploymentIndicatorState;
}

export const DeploymentActions: FC<DeploymentActionsProps> = ({ model, adapter, deployment }) => {
  const servedRef = deployment.kind === 'served' ? deployment.deploymentRef : undefined;
  const servedParts = servedRef ? getPartsFromReference(servedRef) : undefined;
  const { data: servedDeployment } = useModelsGetLatestDeployment(
    servedParts?.workspace ?? '',
    servedParts?.name ?? '',
    { query: { enabled: Boolean(servedParts) } }
  );

  const modelRef: ResourceRef = `${model.workspace}/${model.name}`;

  switch (deployment.kind) {
    case 'served':
      return (
        <Flex gap="density-sm">
          <Button kind="secondary" asChild>
            <Link
              to={getModelDetailRoute(model.workspace, model.name, {
                adapterName: adapter?.name,
                tab: ModelDetailTab.Chat,
              })}
            >
              Chat
            </Link>
          </Button>
          {servedDeployment && (
            <DeleteDeploymentButton model={model} deployment={servedDeployment} />
          )}
        </Flex>
      );
    case 'not-serving': {
      const { status } = deployment.deployment;
      return (
        <Flex gap="density-sm">
          <DeleteDeploymentButton model={model} deployment={deployment.deployment} />
          {!(status && LIVE_STATUSES.has(status)) && (
            <DeployModelCta modelRef={modelRef} label="Redeploy…" />
          )}
        </Flex>
      );
    }
    case 'not-deployed':
      return (
        <DeployModelCta modelRef={modelRef} label={adapter ? 'Deploy base model' : 'Deploy'} />
      );
    default:
      return null;
  }
};
