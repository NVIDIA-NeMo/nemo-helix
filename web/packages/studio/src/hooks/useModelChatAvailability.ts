// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { isServedByBaseModel } from '@nemo/common/src/utils/models';
import { useModelsGetModel } from '@nemo/sdk/generated/platform/models';
import type { Adapter, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useModelDeploymentStatus } from '@studio/hooks/useModelDeploymentStatus';
import { useModelEntityChatStatusWithGrace } from '@studio/hooks/useModelEntityChatStatusWithGrace';
import { useModelIsServed } from '@studio/hooks/useModelIsServed';

interface UseModelChatAvailabilityOptions {
  adapter?: Adapter | null;
}

export function useModelChatAvailability(
  model: ModelEntity | null | undefined,
  options?: UseModelChatAvailabilityOptions
) {
  const adapter = options?.adapter;
  // Only an adapter is served by a deployment of its base. A full-weight or merged
  // fine-tune also carries `base_model`, but it registers its own Model Entity,
  // deployment and provider — resolving its status against the base would report
  // "no active deployment" for the normal case where only the fine-tune is deployed.
  // For an adapter *row*, the model is already the parent, so it is checked directly.
  const baseModelRef =
    adapter || !model || !isServedByBaseModel(model) ? undefined : model.base_model;

  // `base_model` is a `workspace/name` reference, while `useModelsGetModel` takes the
  // two apart. Passing the reference as the name asks for `.../models/default/qwen3`,
  // which 404s — and a base that fails to load is indistinguishable from one that is
  // not deployed, so the adapter reports itself unavailable.
  const baseParts = baseModelRef?.includes('/') ? getPartsFromReference(baseModelRef) : undefined;

  const { data: baseModelEntity, isLoading: isLoadingBaseModel } = useModelsGetModel(
    baseParts?.workspace ?? model?.workspace ?? '',
    baseParts?.name ?? baseModelRef ?? '',
    undefined,
    { query: { enabled: Boolean(baseModelRef), retry: false } }
  );

  const modelForStatus = baseModelRef ? baseModelEntity : model;
  const { status: deploymentStatus, isLoading: isStatusLoading } = useModelDeploymentStatus(
    modelForStatus ?? undefined
  );
  // With an adapter, this asks whether the *adapter* is served, not its base: a READY
  // base deployment that has not loaded the adapter serves the base and nothing else.
  const {
    isServed,
    isLoading: isServedLoading,
    isError: isServedError,
  } = useModelIsServed(modelForStatus, adapter);

  const isLoading = baseModelRef
    ? isLoadingBaseModel || isStatusLoading || isServedLoading
    : isStatusLoading || isServedLoading;

  // Use the original model entity for grace period / api_endpoint checks;
  // deployment status comes from the resolved model (base or self).
  const graceStatus = useModelEntityChatStatusWithGrace(model, {
    adapter,
    deploymentStatus,
    deploymentLoading: isLoading,
  });

  // If the grace/deployment check says enabled but no provider actually serves
  // the model, override to disabled. This catches stale model_providers refs
  // where the model was removed from the provider's served_models.
  const modelChatStatus =
    graceStatus === 'enabled' && !isServedLoading && !isServed ? 'disabled' : graceStatus;

  const isChatAvailable = modelChatStatus === 'enabled';

  /**
   * An adapter whose base is resolved, but which no provider lists among its served
   * models. Distinct from a plain unavailable model: the base deployment may be fine
   * and still coming up to the adapter, so callers give it its own empty state rather
   * than the generic "Chat Unavailable".
   *
   * Excludes a failed provider lookup. Provider queries do not retry, so a single 5xx
   * or a stale provider reference yields `isServed: false` without establishing it —
   * and this drives copy that tells the user the base deployment has not loaded the
   * adapter yet, which would then be a confident guess. Those fall through to the
   * generic unavailable state instead.
   */
  const isAdapterUnserved = Boolean(adapter) && !isLoading && !isServedError && !isServed;

  return { modelChatStatus, isChatAvailable, isLoading, isAdapterUnserved };
}
