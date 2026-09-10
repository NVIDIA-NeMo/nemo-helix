// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetDeploymentConfigVersion } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { useModelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import { ModelDeploymentStatus } from '@nemo/sdk/generated/platform/schema';

/**
 * What the fine-tuning form needs to know about a base model before training an
 * adapter against it.
 *
 * - `none` — nothing serves this model. The adapter would be unservable.
 * - `serving-lora` — a live deployment with `lora_enabled`. Nothing to do.
 * - `serving-without-lora` — a live deployment that will refuse the adapter.
 * - `unavailable` — a deployment exists but is in a terminal failed state.
 */
export type BaseModelDeploymentState =
  | 'none'
  | 'serving-lora'
  | 'serving-without-lora'
  | 'unavailable';

const LIVE_STATUSES: ReadonlySet<ModelDeploymentStatus> = new Set<ModelDeploymentStatus>([
  ModelDeploymentStatus.CREATED,
  ModelDeploymentStatus.PENDING,
  ModelDeploymentStatus.READY,
]);

export interface BaseModelDeploymentReadiness {
  state: BaseModelDeploymentState;
  /** Name of the resolved deployment, for copy that names what is already serving. */
  deploymentName: string | null;
  status: ModelDeploymentStatus | null;
  isLoading: boolean;
}

/**
 * Resolve whether `modelRef` ("workspace/name") is already serving LoRA adapters.
 *
 * `lora_enabled` lives on the deployment **config**, not the deployment, so this
 * adds a hop that `useModelDeploymentStatus` does not make. It reads the pinned
 * `config_version` rather than the latest, because configs are immutable and
 * versioned — the latest version may have been edited after this deployment
 * started, and would then describe something that is not running.
 */
export function useBaseModelDeploymentReadiness(
  modelRef: string | undefined,
  options?: { enabled?: boolean }
): BaseModelDeploymentReadiness {
  const enabled = options?.enabled !== false && Boolean(modelRef);
  const parts = modelRef ? getPartsFromReference(modelRef) : null;

  const { data: model, isLoading: isLoadingModel } = useModelsGetModel(
    parts?.workspace ?? '',
    parts?.name ?? '',
    undefined,
    { query: { enabled: enabled && Boolean(parts?.name), retry: false } }
  );

  const providerId = model?.model_providers?.[0];
  const providerParts = providerId ? getPartsFromReference(providerId) : null;

  const { data: provider, isLoading: isLoadingProvider } = useModelsGetProvider(
    providerParts?.workspace ?? parts?.workspace ?? '',
    providerParts?.name ?? '',
    { query: { enabled: enabled && Boolean(providerParts?.name), retry: false } }
  );

  const deploymentParts = provider?.model_deployment_id
    ? getPartsFromReference(provider.model_deployment_id)
    : null;

  const { data: deployment, isLoading: isLoadingDeployment } = useModelsGetLatestDeployment(
    deploymentParts?.workspace ?? parts?.workspace ?? '',
    deploymentParts?.name ?? '',
    { query: { enabled: enabled && Boolean(deploymentParts?.name), retry: false } }
  );

  const { data: config, isLoading: isLoadingConfig } = useModelsGetDeploymentConfigVersion(
    deploymentParts?.workspace ?? parts?.workspace ?? '',
    deployment?.config ?? '',
    deployment?.config_version !== undefined ? String(deployment.config_version) : '',
    { query: { enabled: enabled && Boolean(deployment?.config), retry: false } }
  );

  const isLoading =
    enabled &&
    (isLoadingModel ||
      (Boolean(providerParts?.name) && isLoadingProvider) ||
      (Boolean(deploymentParts?.name) && isLoadingDeployment) ||
      (Boolean(deployment?.config) && isLoadingConfig));

  const status: ModelDeploymentStatus | null = deployment?.status ?? null;

  let state: BaseModelDeploymentState = 'none';
  if (status && LIVE_STATUSES.has(status)) {
    state = config?.model_spec?.lora_enabled ? 'serving-lora' : 'serving-without-lora';
  } else if (status) {
    state = 'unavailable';
  }

  return {
    state,
    deploymentName: deployment?.name ?? null,
    status,
    isLoading,
  };
}
