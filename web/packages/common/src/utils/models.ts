// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type Adapter,
  FinetuningType,
  ModelDeploymentStatus,
  type ModelEntity,
} from '@nemo/sdk/generated/platform/schema';

import { buildWorkspaceGroup, type ModelWorkspaceGroup } from '../api/models/useModels';
import { getURNFromNamedEntityRef } from '../namedEntity';
import type { ResourceRef } from '../types';

const DEFAULT_WORKSPACE = 'default';

/**
 * Groups a flat list of models into {@link ModelWorkspaceGroup}s keyed by
 * `model.workspace` (falling back to `'default'`). Empty input yields `[]`.
 *
 * Pass `sort: true` to sort the resulting groups alphabetically by workspace
 * name. Models within each group preserve their input order.
 */
export const groupModelsByWorkspace = (
  models: ModelEntity[],
  { sort = false }: { sort?: boolean } = {}
): ModelWorkspaceGroup[] => {
  if (!models.length) return [];
  const byWorkspace = models.reduce<Record<string, ModelEntity[]>>((acc, model) => {
    const ws = model.workspace ?? DEFAULT_WORKSPACE;
    (acc[ws] ??= []).push(model);
    return acc;
  }, {});
  const entries = Object.entries(byWorkspace);
  if (sort) entries.sort(([a], [b]) => a.localeCompare(b));
  return entries.map(([ws, ms]) => buildWorkspaceGroup(ws, ms));
};

export const hasModelProvider = (model: ModelEntity): boolean =>
  Array.isArray(model.model_providers) && model.model_providers.length > 0;

/**
 * Returns true if the model is a base model, false otherwise. This is determined by
 * checking if the model has a base_model property.
 *
 * @param model - The model to check.
 * @returns True if the model is a base model, false otherwise.
 */
export const isBaseModel = (model: ModelEntity) => {
  return model.base_model === undefined;
};

/**
 * Given a ModelEntity, return the base model URN
 * @param model - The model to get the base model for
 * @returns The base model URN
 */
export const getBaseModelURN = (model: ModelEntity) => {
  if (typeof model.base_model === 'string') {
    return model.base_model;
  }
  return getURNFromNamedEntityRef(model.base_model);
};

/**
 * The model name a chat request puts on the wire, for the gateway's model-entity
 * route (`/workspaces/{workspace}/model/{name}/-/v1/...`).
 *
 * For a plain model this is just its name. For a LoRA adapter it is the composite
 *
 *     {base_name}&adapters/{adapter_workspace}/{adapter_name}
 *
 * The base segment is the **bare** entity name, not `{workspace}/{name}`: the
 * gateway splits the composite on `&adapters/` and looks up the VirtualModel by
 * `(workspace, base_name)`, with the workspace taken from the URL. The adapter
 * segment carries its own workspace because an adapter may live in a different one
 * from its base.
 *
 * The gateway resolves the composite through the base model's VirtualModel — so an
 * adapter request inherits that VM's middleware (guardrails, Switchyard routing,
 * translate) — and rewrites the body to the backend's own served name on the way
 * out. That name is therefore the gateway's concern, not the client's.
 *
 * @param model - The base model entity being served
 * @param adapter - The adapter to target, if any
 * @returns The value for the request's `model` field and URL name segment
 */
export const toInferenceModelName = (model: ModelEntity, adapter?: Adapter | null): string =>
  adapter ? `${model.name}&adapters/${adapter.workspace}/${adapter.name}` : model.name;

/**
 * Platform-side model entity id, as it appears in a provider's `served_models`.
 *
 * This is an **identity, not a wire value** — use {@link toInferenceModelName} for
 * the latter. Its use is to find the provider entry for a model or adapter, which
 * answers whether anything actually serves it.
 *
 * A LoRA adapter has no standalone Model Entity, so the platform addresses it with a
 * composite id built by the provider reconciler; this is {@link toInferenceModelName}
 * qualified by the base model's workspace.
 *
 * @param model - The base model entity being served
 * @param adapter - The adapter to target, if any
 * @returns The `model_entity_id` to match against `provider.served_models`
 */
export const toInferenceModelEntityId = (model: ModelEntity, adapter?: Adapter | null): string =>
  `${model.workspace}/${toInferenceModelName(model, adapter)}`;

/**
 * Build a model config for evaluation related types like targets and metrics
 * @param model_id - The model URN to build the config for
 * @returns The model config
 */
export const buildModelConfig = (
  model_id: ResourceRef,
  platformUrl: string,
  isChat: boolean = true
) => {
  return {
    api_endpoint: {
      url: isChat ? `${platformUrl}/v1/chat/completions` : `${platformUrl}/v1/completions`,
      model_id,
      format: 'nim' as const,
    },
  };
};

export const MODEL_DEPLOYMENT_GRACE_PERIOD_MS = 5 * 60 * 1000; // 5 minutes

// Matches timezone indicators: Z, +HH:MM, -HH:MM, +HHMM, -HHMM
const TIMEZONE_REGEX = /(Z|[+-]\d{2}:?\d{2})$/;

/**
 * Check if a model was created within the deployment grace period.
 * Models need time after creation for the backend to deploy them and populate the artifact.
 *
 * Only meaningful for models that are waiting on a deployment. A model that already
 * carries an `api_endpoint.url` has no artifact pending, so callers exempt it — see
 * {@link getModelEntityChatStatus}.
 */
const isWithinDeploymentGracePeriod = (model: ModelEntity) => {
  if (!model.created_at) {
    return false;
  }
  // Ensure the timestamp is parsed as UTC if no timezone indicator is present
  const hasTimezone = TIMEZONE_REGEX.test(model.created_at);
  const timestamp = hasTimezone ? model.created_at : `${model.created_at}Z`;
  const createdAt = new Date(timestamp).getTime();
  const now = Date.now();
  return now - createdAt < MODEL_DEPLOYMENT_GRACE_PERIOD_MS;
};

/**
 * Finetuning types that emit an **adapter** rather than a set of weights.
 *
 * A model trained this way has nothing to serve on its own: inference runs on a
 * deployment of its base model, with the adapter loaded alongside. Everything
 * else in {@link FinetuningType} — `all_weights`, `lora_merged`, the partial-
 * freeze variants, and every preference-optimisation type — produces a complete
 * model that owns its own deployment.
 *
 * Listed explicitly rather than matched on a `lora` prefix: `lora_merged` starts
 * with it and is the opposite case, and `prompt_tuning` / `p_tuning` / `soft_prompt`
 * do not contain it at all.
 */
const ADAPTER_FINETUNING_TYPES: ReadonlySet<FinetuningType> = new Set([
  FinetuningType.lora,
  FinetuningType.qlora,
  FinetuningType.adalora,
  FinetuningType.dora,
  FinetuningType.lora_plus,
  FinetuningType.prompt_tuning,
  FinetuningType.prefix_tuning,
  FinetuningType.p_tuning,
  FinetuningType.p_tuning_v2,
  FinetuningType.soft_prompt,
]);

/**
 * Whether inference for this model runs on a deployment of its **base model**.
 *
 * `base_model` alone does not answer this. It records what the model was derived
 * from, and a full-weight fine-tune sets it just as an adapter does — but that run
 * registers a new Model Entity with its own weights, its own deployment and its own
 * provider. Reading `base_model` as "served by the base" reports such a model as
 * having no active deployment whenever the base happens not to be deployed, which is
 * the normal state after a full-weight run.
 *
 * `finetuning_type` is what distinguishes them. Absent, the model is not a fine-tune
 * at all and is served on its own.
 */
export function isServedByBaseModel(model: ModelEntity): boolean {
  if (!model.base_model) return false;
  return Boolean(model.finetuning_type && ADAPTER_FINETUNING_TYPES.has(model.finetuning_type));
}

export type ModelChatStatus = 'enabled' | 'disabled' | 'pending';

const DEPLOYMENT_TERMINAL_DISABLED: ReadonlySet<ModelDeploymentStatus> = new Set([
  ModelDeploymentStatus.ERROR,
  ModelDeploymentStatus.DELETED,
  ModelDeploymentStatus.LOST,
  ModelDeploymentStatus.DELETING,
]);

export type GetModelEntityChatStatusOptions = {
  /**
   * When the UI is scoped to an adapter row, chat requires a READY deployment
   * (same as entities with `base_model`).
   */
  adapter?: Adapter | null;
  /** Provider/deployment chain is still loading */
  deploymentLoading?: boolean;
  /**
   * Resolved {@link ModelDeploymentStatus} for the entity that backs inference
   * (e.g. base model when `model.base_model` is set — callers resolve which entity to query).
   * - Omit when no deployment query ran (list/playground): legacy optimistic behavior.
   * - `null` when no deployment exists.
   */
  deploymentStatus?: ModelDeploymentStatus | null;
};

/**
 * Whether chat can run for this model in the current UI context.
 *
 * Order of evaluation:
 * 1. `deploymentLoading` → `'pending'`
 * 2. Creation grace period → `'pending'`, **unless** the entity already has an
 *    `api_endpoint.url` (endpoint-backed models have nothing to deploy).
 * 3. Entities with `base_model` or an **adapter** row need a READY deployment; `api_endpoint` alone is not enough.
 * 4. Standalone entities: `api_endpoint.url`, READY deployment, or (if deployment unknown) optimistic `'enabled'`.
 */
export function getModelEntityChatStatus(
  model: ModelEntity,
  options?: GetModelEntityChatStatusOptions
): ModelChatStatus {
  const { adapter, deploymentLoading, deploymentStatus } = options ?? {};

  const hasApiEndpoint = Boolean(model.api_endpoint?.url);

  // The grace period covers the gap between an entity being created and the backend
  // populating its deployment artifact. A model autodiscovered from an external
  // provider (build.nvidia.com, any OpenAI-compatible host) has its api_endpoint set
  // at creation and is serveable immediately, so applying the timer there reports a
  // deployment that does not exist and never will.
  if (deploymentLoading || (!hasApiEndpoint && isWithinDeploymentGracePeriod(model))) {
    return 'pending';
  }

  const needsReadyDeployment = Boolean(adapter) || Boolean(model.base_model);

  if (needsReadyDeployment) {
    if (deploymentStatus === undefined || deploymentStatus === ModelDeploymentStatus.READY) {
      return 'enabled';
    }
    if (deploymentStatus === null || DEPLOYMENT_TERMINAL_DISABLED.has(deploymentStatus)) {
      return 'disabled';
    }
    return 'pending';
  }

  if (
    hasApiEndpoint ||
    deploymentStatus === undefined ||
    deploymentStatus === ModelDeploymentStatus.READY
  ) {
    return 'enabled';
  }
  if (deploymentStatus === null || DEPLOYMENT_TERMINAL_DISABLED.has(deploymentStatus)) {
    return 'disabled';
  }
  return 'pending';
}

/**
 * Returns a boolean based on a search string and model. Checks if the model name, namespace, or URN matches the search string.
 * Returns true if the model matches the search string, false otherwise.
 */
export const filterModel = (model: ModelEntity, search?: string) => {
  const workspace = model.workspace;
  return !(
    search &&
    !model.name?.toLowerCase().includes(search.toLowerCase()) &&
    !workspace?.toLowerCase().includes(search.toLowerCase()) &&
    !getURNFromNamedEntityRef(model)?.toLowerCase().includes(search.toLowerCase())
  );
};
