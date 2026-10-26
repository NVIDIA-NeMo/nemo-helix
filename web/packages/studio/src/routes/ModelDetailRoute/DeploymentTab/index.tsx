// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Banner, Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import { Empty } from '@studio/components/Empty';
import { Loading } from '@studio/components/Layouts/Loading';
import { LIVE_STATUSES } from '@studio/hooks/useBaseModelDeploymentReadiness';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import { ServedView } from '@studio/routes/ModelDetailRoute/DeploymentTab/ServedView';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getWorkspaceDeploymentDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

const DeploymentsIcon = ENTITY_ICONS.deployments;

export interface DeploymentTabProps {
  model: ModelEntity;
  adapter: Adapter | null;
  /** The base model's deployment state: an adapter is served through its model's deployment. */
  deployment: DeploymentIndicatorState;
}

export const DeploymentTab: FC<DeploymentTabProps> = ({ model, adapter, deployment }) => {
  switch (deployment.kind) {
    case 'loading':
      return <Loading description="Loading deployment..." />;

    case 'served':
      return (
        <ServedView
          model={model}
          adapter={adapter}
          providerRef={deployment.providerRef}
          deploymentRef={deployment.deploymentRef}
        />
      );

    case 'not-serving': {
      const { deployment: unserved } = deployment;
      const isStarting = Boolean(unserved.status && LIVE_STATUSES.has(unserved.status));
      return (
        <Stack gap="density-3xl" className="w-full">
          {!isStarting && (
            <Banner kind="inline" status="error">
              {unserved.status_message || 'The deployment stopped before it became ready.'}
            </Banner>
          )}
          <Grid colMinWidth="220px" gap="density-lg">
            <LineageCard
              label="Deployment"
              value={unserved.name}
              to={getWorkspaceDeploymentDetailsRoute(unserved.workspace, unserved.name)}
              badge={<StatusBadge status={unserved.status} />}
              note={unserved.created_at && formatAbsoluteTimestamp(unserved.created_at)}
            />
          </Grid>
          {isStarting && unserved.status_message && (
            <Text kind="body/regular/sm" className="text-secondary">
              {unserved.status_message}
            </Text>
          )}
        </Stack>
      );
    }

    case 'not-deployed':
      return (
        <Flex direction="col" justify="center" className="flex-1">
          <Empty
            icon={<DeploymentsIcon className="size-12" />}
            title={adapter ? 'Base model not deployed' : 'Not deployed'}
            description={
              adapter
                ? `Deploy ${model.name} to serve this adapter.`
                : 'Deploy this model to chat with it and serve inference.'
            }
          />
        </Flex>
      );

    default:
      return (
        <ErrorMessage
          header="Couldn't check the deployment"
          message="A provider for this model couldn't be read, so whether it is served is unknown."
        />
      );
  }
};
