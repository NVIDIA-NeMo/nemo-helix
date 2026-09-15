// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { toInferenceModelEntityId } from '@nemo/common/src/utils/models';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import type {
  Adapter,
  ModelDeploymentStatus,
  ModelEntity,
} from '@nemo/sdk/generated/platform/schema';
import { useServedModel } from '@studio/hooks/useServedModel';

export type DeploymentIndicatorState =
  /** Still resolving providers or the deployment. */
  | { kind: 'loading' }
  /**
   * A provider lists this exact id in its `served_models`, so inference would
   * route. `hasDeployment` says whether the serving provider names a backing
   * deployment: false means an external provider (such as `default/build`) that
   * is reachable but has no deployment to report on, whereas true with no
   * `status` means a deployment exists but could not be read.
   */
  | {
      kind: 'served';
      hasDeployment: boolean;
      status?: ModelDeploymentStatus;
      statusMessage?: string;
      providerRef?: string;
    }
  /** Adapter only: the base model is served, but this adapter is not loaded. */
  | { kind: 'adapter-not-loaded' }
  /** Nothing serves this id, and (for an adapter) nothing serves its base either. */
  | { kind: 'not-deployed' }
  /**
   * A provider could not be fetched, so whether this is served is genuinely
   * unknown. Distinct from `not-deployed`, which asserts nothing serves it.
   */
  | { kind: 'unknown' };

/**
 * Resolve whether a model entity — or a specific adapter of it — is actually being
 * served, and with what deployment status.
 *
 * Matches against each provider's `served_models`, which the platform's provider
 * reconciler builds from what the running backend reports. That is the same table
 * Inference Gateway routes on, so a `served` result means a request would resolve,
 * rather than merely that a deployment was requested with LoRA enabled.
 *
 * @param model - The model entity that owns the row. For an adapter subrow this is
 *   the *parent* model, not the flattened row.
 * @param adapter - The adapter to check, when the row is an adapter subrow.
 */
export function useModelDeploymentIndicator(
  model: ModelEntity | null | undefined,
  adapter?: Adapter | null
): DeploymentIndicatorState {
  const targetId = model ? toInferenceModelEntityId(model, adapter) : '';
  const baseId = model ? toInferenceModelEntityId(model) : '';

  const {
    servedModel,
    provider,
    providerRef,
    isLoading: isTargetLoading,
    isError: isTargetError,
  } = useServedModel(model, targetId);

  // Only needed to tell "base is up but this adapter isn't loaded" from "nothing is
  // deployed". Passing '' disables the lookup; for a top-level row targetId === baseId
  // so it never runs at all.
  const needsBaseProbe = Boolean(adapter) && !isTargetLoading && !servedModel;
  const {
    servedModel: baseServedModel,
    isLoading: isBaseLoading,
    isError: isBaseError,
  } = useServedModel(model, needsBaseProbe ? baseId : '');

  const deploymentParts = provider?.model_deployment_id
    ? getPartsFromReference(provider.model_deployment_id)
    : null;

  const { data: deployment, isLoading: isDeploymentLoading } = useModelsGetLatestDeployment(
    deploymentParts?.workspace ?? '',
    deploymentParts?.name ?? '',
    { query: { enabled: Boolean(deploymentParts?.name), retry: false } }
  );

  if (isTargetLoading) return { kind: 'loading' };

  if (servedModel) {
    if (deploymentParts && isDeploymentLoading) return { kind: 'loading' };
    return {
      kind: 'served',
      // Whether a deployment is *expected*, which is not the same as having read
      // one. Callers need the difference to avoid reporting a provider whose
      // deployment request failed as though it had no deployment at all.
      hasDeployment: Boolean(deploymentParts),
      status: deployment?.status,
      statusMessage: deployment?.status_message,
      providerRef,
    };
  }

  // A provider we could not read might be the one serving this id, so nothing below
  // may assert otherwise. This has to precede the base probe: with several
  // providers, one readable provider can serve the base while the failed one serves
  // the adapter, and matching the base first would claim "Not served" on evidence we
  // never gathered.
  if (isTargetError) return { kind: 'unknown' };

  if (needsBaseProbe) {
    if (isBaseLoading) return { kind: 'loading' };
    if (baseServedModel) return { kind: 'adapter-not-loaded' };
    if (isBaseError) return { kind: 'unknown' };
  }

  return { kind: 'not-deployed' };
}
