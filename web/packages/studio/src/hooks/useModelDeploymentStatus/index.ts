// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import { useModelsGetProvider } from '@nemo/sdk/generated/platform/model-providers';
import type { ModelDeploymentStatus, ModelEntity } from '@nemo/sdk/generated/platform/schema';

/**
 * Resolves the deployment status for a model entity by walking the chain:
 * model_providers → ModelProvider → ModelDeployment.
 */
export function useModelDeploymentStatus(model: ModelEntity | undefined) {
  const workspace = model?.workspace ?? '';
  const providerIds = model?.model_providers;
  const hasProviders = Boolean(providerIds?.length);

  const firstProviderId = providerIds?.[0];
  const providerParts = firstProviderId ? getPartsFromReference(firstProviderId) : null;

  const { data: provider, isLoading: isLoadingProvider } = useModelsGetProvider(
    providerParts?.workspace ?? workspace,
    providerParts?.name ?? '',
    { query: { enabled: hasProviders && Boolean(providerParts?.name), retry: false } }
  );

  const deploymentParts = provider?.model_deployment_id
    ? getPartsFromReference(provider.model_deployment_id)
    : null;

  const { data: deployment, isLoading: isLoadingDeployment } = useModelsGetLatestDeployment(
    deploymentParts?.workspace ?? workspace,
    deploymentParts?.name ?? '',
    { query: { enabled: hasProviders && Boolean(deploymentParts?.name), retry: false } }
  );

  const isLoading = hasProviders && (isLoadingProvider || isLoadingDeployment);
  const status: ModelDeploymentStatus | null = deployment?.status ?? null;

  return {
    /** The resolved deployment status, or null if no deployment found */
    status,
    /** Whether the provider/deployment chain is still loading */
    isLoading,
  };
}
