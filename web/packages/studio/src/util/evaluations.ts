// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import { snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type { EvaluateJob, Model } from '@nemo/sdk/generated/evals/schema';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';

/**
 * Extracts the model name from a evaluation job.
 * @param row - The evaluation job
 * @returns The model name or 'N/A' if not found
 */
export const getModelName = (row?: EvaluateJob): string => {
  const target = row?.spec?.target;
  if (target && 'name' in target && typeof target.name === 'string') {
    return target.name;
  }
  return 'N/A';
};

/**
 * Converts snake_case metric/score names to Title Case and removes 'Score' suffix.
 * @param name - The name to prettify
 * @returns The prettified name
 */
export const prettifyName = (name: string): string => {
  const formattedName = snakeCaseToTitleCase(name);
  return formattedName.endsWith('Score')
    ? formattedName.replace(/Score$/, '').trim()
    : formattedName;
};

/**
 * Parse a value produced by an EvaluationModelItem back into its components.
 *
 * Model values are plain URNs (`workspace/name`).
 * Adapter values use a `::` delimiter (`workspace/name::adapterName`).
 */
export const parseEvaluationModelValue = (
  value: string
): { modelUrn: string; adapterName: string | null } => {
  const idx = value.indexOf('::');
  if (idx === -1) return { modelUrn: value, adapterName: null };
  return { modelUrn: value.substring(0, idx), adapterName: value.substring(idx + 2) };
};

export type BuildModelPayloadResult =
  | { ok: true; payload: Model | string }
  | { ok: false; error: string };

/**
 * Build the model payload for a metric evaluation job submission.
 *
 * For base models the value string is forwarded as a ModelRef.
 * For adapters an inline Model is built using the parent model's
 * provider proxy URL so the inference gateway preserves the adapter name.
 */
export const buildModelPayload = (
  modelValue: string,
  models: ModelEntity[],
  origin: string
): BuildModelPayloadResult => {
  const { modelUrn, adapterName } = parseEvaluationModelValue(modelValue);

  if (!adapterName) {
    return { ok: true, payload: modelValue };
  }

  const parentModel = models.find((m) => getURNFromNamedEntityRef(m) === modelUrn);
  if (!parentModel) {
    return { ok: false, error: 'Selected model not found or still loading.' };
  }

  const providerRef = parentModel.model_providers?.[0];
  if (!providerRef) {
    return { ok: false, error: 'Selected model has no provider configured for adapter inference.' };
  }

  const providerName = providerRef.includes('/') ? providerRef.split('/').pop()! : providerRef;

  return {
    ok: true,
    payload: {
      url: `${origin}/apis/inference-gateway/v2/workspaces/${parentModel.workspace}/provider/${providerName}/-/v1`,
      name: adapterName,
    },
  };
};
