// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { toInferenceModelEntityId } from '@nemo/common/src/utils/models';
import { useProviderReady } from '@nemo/sdk/generated/platform/inference-gateway';
import { useModelsGetDeploymentConfigVersion } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { useModelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { Grid, Stack } from '@nvidia/foundations-react-core';
import { CopyButton } from '@studio/components/CopyButton';
import { DeploymentStatusBadge } from '@studio/components/dataViews/CustomModelsDataView/DeploymentStatusBadge';
import {
  getConfigFields,
  getConfigTiles,
} from '@studio/routes/ModelDetailRoute/DeploymentTab/configFields';
import { GatewayPanel } from '@studio/routes/ModelDetailRoute/DeploymentTab/GatewayPanel';
import { AdaptersSection } from '@studio/routes/ModelDetailRoute/DetailsTab/AdaptersSection';
import { DetailPanel } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailPanel';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getWorkspaceDeploymentDetailsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

export interface ServedViewProps {
  model: ModelEntity;
  adapter: Adapter | null;
  providerRef?: string;
  deploymentRef?: string;
}

export const ServedView: FC<ServedViewProps> = ({ model, adapter, providerRef, deploymentRef }) => {
  const deploymentParts = deploymentRef ? getPartsFromReference(deploymentRef) : undefined;
  const providerParts = providerRef ? getPartsFromReference(providerRef) : undefined;

  const { data: deployment, isLoading: isDeploymentLoading } = useModelsGetLatestDeployment(
    deploymentParts?.workspace ?? '',
    deploymentParts?.name ?? '',
    { query: { enabled: Boolean(deploymentParts) } }
  );
  const { data: config } = useModelsGetDeploymentConfigVersion(
    deployment?.workspace ?? '',
    deployment?.config ?? '',
    deployment ? String(deployment.config_version) : '',
    { query: { enabled: Boolean(deployment) } }
  );
  const { data: provider, isLoading: isProviderLoading } = useModelsGetProvider(
    providerParts?.workspace ?? '',
    providerParts?.name ?? '',
    { query: { enabled: Boolean(providerParts) } }
  );
  const { isSuccess: isRoutable, isLoading: isReadyLoading } = useProviderReady(
    providerParts?.workspace ?? '',
    providerParts?.name ?? '',
    {
      query: {
        enabled: Boolean(providerParts),
        retry: false,
        refetchInterval: (query) =>
          query.state.status === 'success' ? false : JOB_POLLING_INTERVAL_MS,
      },
    }
  );

  const servedIds = new Set(provider?.served_models?.map((served) => served.model_entity_id));
  const isLoaded = (candidate: Adapter) =>
    servedIds.has(toInferenceModelEntityId(model, candidate));
  const adapters = [...(model.adapters ?? [])].sort(
    (a, b) => Number(b.name === adapter?.name) - Number(a.name === adapter?.name)
  );
  const showAdapters = Boolean(config?.model_spec.lora_enabled) && adapters.length > 0;

  return (
    <Stack gap="density-3xl" className="w-full pb-density-2xl">
      <Grid colMinWidth="220px" gap="density-lg">
        {deploymentParts && (
          <LineageCard
            label="Deployment"
            loading={isDeploymentLoading}
            value={deploymentParts.name}
            to={getWorkspaceDeploymentDetailsRoute(deploymentParts.workspace, deploymentParts.name)}
            badge={<StatusBadge status={deployment?.status} />}
            note={deployment?.created_at && formatAbsoluteTimestamp(deployment.created_at)}
          />
        )}
        {providerParts && (
          <LineageCard
            label="Provider"
            loading={isProviderLoading}
            value={providerParts.name}
            badge={<StatusBadge status={provider?.status} />}
            note={deploymentParts ? undefined : 'External provider'}
          />
        )}
        {providerParts && (
          <LineageCard
            label="Gateway"
            loading={isReadyLoading}
            value={isRoutable ? 'Routable' : 'Not routable yet'}
            note={
              isRoutable ? 'Chat requests reach this model' : "The gateway hasn't registered it"
            }
          />
        )}
        {provider?.host_url && (
          <LineageCard
            label="Endpoint"
            value={provider.host_url}
            badge={
              <CopyButton text={provider.host_url} kind="tertiary" aria-label="Copy endpoint" />
            }
          />
        )}
      </Grid>

      <Grid cols={{ base: 1, lg: 2 }} gap="density-xl">
        {config && (
          <DetailPanel
            heading="Configuration"
            description={config.description}
            tiles={getConfigTiles(config)}
            fields={getConfigFields(config)}
          />
        )}
        <GatewayPanel model={model} />
      </Grid>

      {showAdapters && (
        <AdaptersSection
          workspace={model.workspace}
          modelName={model.name}
          adapters={adapters}
          description={`${adapters.filter(isLoaded).length} of ${adapters.length} loaded on this deployment.`}
          renderStatus={(candidate) => (
            <DeploymentStatusBadge
              state={
                isLoaded(candidate)
                  ? {
                      kind: 'served',
                      hasDeployment: Boolean(deploymentRef),
                      status: deployment?.status,
                    }
                  : { kind: 'adapter-not-loaded' }
              }
              isAdapter
            />
          )}
        />
      )}
    </Stack>
  );
};
