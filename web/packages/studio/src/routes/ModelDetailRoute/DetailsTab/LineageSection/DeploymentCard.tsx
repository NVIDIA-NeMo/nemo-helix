// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getWorkspaceDeploymentDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

const getDeploymentCard = (
  state: DeploymentIndicatorState,
  isAdapter: boolean
): { value: string; note?: string; to?: string } => {
  switch (state.kind) {
    case 'served': {
      if (state.deploymentRef) {
        const { workspace, name } = getPartsFromReference(state.deploymentRef);
        return { value: name, to: getWorkspaceDeploymentDetailsRoute(workspace, name) };
      }
      return {
        value: state.providerRef ? getPartsFromReference(state.providerRef).name : 'External',
        note: 'Served by an external provider',
      };
    }
    case 'not-serving': {
      const { workspace, name, status_message: statusMessage } = state.deployment;
      return {
        value: name,
        to: getWorkspaceDeploymentDetailsRoute(workspace, name),
        note: statusMessage || 'Not serving yet',
      };
    }
    case 'adapter-not-loaded':
      return { value: 'Not loaded', note: "The model's deployment hasn't loaded this adapter" };
    case 'not-deployed':
      return isAdapter
        ? { value: 'Base not deployed', note: 'Deploy the base model to serve this adapter' }
        : { value: 'Not deployed', note: 'Deploy it to chat and serve inference' };
    default:
      return { value: 'Unknown', note: "Couldn't check where this is served" };
  }
};

export interface DeploymentCardProps {
  deployment: DeploymentIndicatorState;
  /** Where the card leads; defaults to the serving deployment's own page. */
  to?: string;
  isAdapter: boolean;
}

export const DeploymentCard: FC<DeploymentCardProps> = ({ deployment, to, isAdapter }) => {
  const card = getDeploymentCard(deployment, isAdapter);
  return (
    <LineageCard
      label="Deployment"
      loading={deployment.kind === 'loading'}
      value={card.value}
      to={to ?? card.to}
      badge={<DeploymentStatusBadge state={deployment} isAdapter={isAdapter} hideUnknown />}
      note={card.note}
    />
  );
};
