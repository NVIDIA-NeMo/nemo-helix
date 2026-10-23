// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useModelsListDeploymentConfigs } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsListDeployments } from '@nemo/sdk/generated/platform/model-deployments';
import { getModelsGetModelQueryKey } from '@nemo/sdk/generated/platform/models';
import {
  type ModelDeployment,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { LIVE_STATUSES } from '@studio/hooks/useBaseModelDeploymentReadiness';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';

const isStarting = (status?: ModelDeploymentStatus) =>
  status === ModelDeploymentStatus.CREATED || status === ModelDeploymentStatus.PENDING;

/**
 * The newest deployment of `model` that isn't being deleted, found through the configs
 * bound to it. Unlike the model's providers, this sees a deployment that is still starting
 * or that failed.
 *
 * Polls while the deployment starts, and refreshes the model until it is served: the model
 * only lists the deployment's provider once the provider is discovered.
 */
export const useLatestModelDeployment = (
  model: ModelEntity | undefined,
  { isServed }: { isServed: boolean }
): { deployment: ModelDeployment | undefined; isLoading: boolean } => {
  const workspace = model?.workspace ?? '';
  const name = model?.name ?? '';

  const { data: configs, isLoading: isConfigsLoading } = useModelsListDeploymentConfigs(
    workspace,
    { filter: { model_entity_id: `${workspace}/${name}` }, page_size: 100 },
    { query: { enabled: Boolean(model) } }
  );
  const configNames = configs?.data.map((config) => config.name) ?? [];
  const { data: deployments, isLoading: isDeploymentsLoading } = useModelsListDeployments(
    workspace,
    { filter: { config: { $in: configNames } }, sort: '-created_at', page_size: 100 },
    {
      query: {
        enabled: configNames.length > 0,
        refetchInterval: (query) =>
          isStarting(query.state.data?.data[0]?.status) ? JOB_POLLING_INTERVAL_MS : false,
      },
    }
  );
  const deployment = deployments?.data.find(
    (candidate) =>
      candidate.status !== ModelDeploymentStatus.DELETING &&
      candidate.status !== ModelDeploymentStatus.DELETED
  );

  const queryClient = useQueryClient();
  const isWaitingToServe =
    !isServed && Boolean(deployment?.status && LIVE_STATUSES.has(deployment.status));
  useEffect(() => {
    if (!isWaitingToServe) return;
    const interval = setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: getModelsGetModelQueryKey(workspace, name) });
    }, JOB_POLLING_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isWaitingToServe, workspace, name, queryClient]);

  return { deployment, isLoading: isConfigsLoading || isDeploymentsLoading };
};
