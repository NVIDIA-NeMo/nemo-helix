// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { toCopyName } from '@nemo/common/src/utils/entityName';
import type {
  CreateJob as DataDesignerJob,
  CreateJobRequest as DataDesignerJobRequest,
} from '@nemo/sdk/generated/data-designer/schema';
import type { ModelEntity, ModelProvider } from '@nemo/sdk/generated/platform/schema';

/** Model option for the form: entity plus display name used in job requests */
export type DataDesignerModelOption = ModelEntity & { served_model_name: string };

/**
 * Resolve workspace and model name for chat/completion calls.
 * If modelRef contains "/", parses as "workspace/name"; otherwise uses fallbackWorkspace and modelRef as name.
 */
export function getWorkspaceAndModel(
  modelRef: string,
  fallbackWorkspace: string
): { workspace: string; name: string } {
  if (modelRef.includes('/')) {
    const parts = getPartsFromReference(modelRef);
    return { workspace: parts.workspace, name: parts.name };
  }
  return { workspace: fallbackWorkspace, name: modelRef };
}

/**
 * Build model options for ModelSelect from the list-providers response.
 * Each provider's served_models becomes an option keyed by model_entity_id; dedupes by id (keeps first served_model_name).
 */
export function modelsFromProviders(providers: ModelProvider[]): DataDesignerModelOption[] {
  const byId = new Map<string, DataDesignerModelOption>();
  for (const provider of providers) {
    const providerRef = `${provider.workspace}/${provider.name}`;
    for (const sm of provider.served_models ?? []) {
      const id = sm.model_entity_id;
      if (!id) continue;
      const [ws, ...nameParts] = id.split('/');
      const name = nameParts.join('/') || sm.served_model_name;
      const workspace = ws ?? provider.workspace;
      const existing = byId.get(id);
      if (existing) {
        const providerSet = new Set(existing.model_providers ?? []);
        providerSet.add(providerRef);
        byId.set(id, { ...existing, model_providers: [...providerSet] });
      } else {
        byId.set(id, {
          id,
          workspace,
          name,
          created_at: '',
          updated_at: '',
          model_providers: [providerRef],
          served_model_name: sm.served_model_name,
        });
      }
    }
  }
  return [...byId.values()];
}

/**
 * Sanitize job request name for API: no spaces (replace with hyphens), collapse runs, trim edges.
 * Entity names typically must not contain spaces.
 */
export function sanitizeJobRequestName(request: DataDesignerJobRequest): DataDesignerJobRequest {
  const name = request.name?.trim();
  if (name == null || name === '') return request;
  const slug = name.replace(/\s+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
  if (slug === '') return request;
  return { ...request, name: slug };
}

/**
 * Parse raw tool-call arguments from the LLM into a DataDesignerJobRequest.
 * Handles both { job_request: ... } and direct spec-shaped payloads.
 */
export function parseToolResponseToJobRequest(rawArgs: string): DataDesignerJobRequest | null {
  let parsed: { job_request?: DataDesignerJobRequest } | DataDesignerJobRequest;
  try {
    parsed = JSON.parse(rawArgs) as
      | { job_request?: DataDesignerJobRequest }
      | DataDesignerJobRequest;
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== 'object') return null;
  if ('job_request' in parsed && parsed.job_request) {
    return parsed.job_request as DataDesignerJobRequest;
  }
  if ('spec' in parsed) return parsed as DataDesignerJobRequest;
  return null;
}

export const PARSE_ERROR_INVALID_JSON = 'Invalid JSON.';
export const PARSE_ERROR_MISSING_CONFIG = 'Invalid structure: missing spec.config.';

/** Result of parsing the JSON editor content */
export type ParseJsonContentResult =
  | { jobRequest: DataDesignerJobRequest; error: null }
  | { jobRequest: null; error: string }
  | { jobRequest: null; error: null };

/**
 * Parse JSON editor content into a job request. Empty content yields null request and no error.
 */
export function parseJsonContentToJobRequest(content: string): ParseJsonContentResult {
  const trimmed = content.trim();
  if (!trimmed) {
    return { jobRequest: null, error: null };
  }
  try {
    JSON.parse(trimmed);
  } catch {
    return { jobRequest: null, error: PARSE_ERROR_INVALID_JSON };
  }
  const parsed = parseToolResponseToJobRequest(trimmed);
  if (!parsed?.spec?.config) {
    return { jobRequest: null, error: PARSE_ERROR_MISSING_CONFIG };
  }
  return { jobRequest: sanitizeJobRequestName(parsed), error: null };
}

/**
 * Build a create-job request from an existing job so it can pre-fill the new-job form.
 * A job's `spec.job_config` is already the `DataDesignerJobConfig` shape a request expects,
 * so cloning is a direct copy of the config plus a "-copy" name. Returns null when the job
 * has no usable config (e.g. a partially-loaded list row).
 */
export function buildClonedJobRequest(job: DataDesignerJob): DataDesignerJobRequest | null {
  const jobConfig = job.spec?.job_config;
  if (!jobConfig?.config) return null;
  return {
    name: job.name ? toCopyName(job.name) : undefined,
    description: job.description,
    spec: jobConfig,
  };
}

/**
 * Narrow an unknown router location state to the cloned job request, if present.
 * Returns null when the state is absent or not shaped like a clone payload.
 */
export function getCloneJobRequestFromState(state: unknown): DataDesignerJobRequest | null {
  if (!state || typeof state !== 'object' || !('cloneJobRequest' in state)) return null;
  const request = (state as { cloneJobRequest: unknown }).cloneJobRequest;
  if (!request || typeof request !== 'object' || !('spec' in request)) return null;
  return request as DataDesignerJobRequest;
}

/**
 * Apply the form's selected model (and provider from ModelEntity.model_providers)
 * to the job request: set model_configs and column model_alias.
 * Uses served_model_name when present (for inference gateway), else formModelRef.
 */
export function applyFormModelToJobRequest(
  jobRequest: DataDesignerJobRequest,
  formModelRef: string,
  provider: string,
  servedModelName: string
): DataDesignerJobRequest {
  const config = jobRequest.spec?.config;
  if (!config?.model_configs?.length) return jobRequest;

  const firstModelConfig = config.model_configs[0];
  const columnsWithFormModel = config.columns.map((col) => {
    const hasModelAlias = 'model_alias' in col && col.model_alias;
    return hasModelAlias ? { ...col, model_alias: formModelRef } : col;
  });

  return {
    ...jobRequest,
    spec: {
      ...jobRequest.spec,
      config: {
        ...config,
        columns: columnsWithFormModel,
        model_configs: [
          {
            ...firstModelConfig,
            alias: formModelRef,
            model: servedModelName,
            provider,
          },
        ],
      },
    },
  };
}
