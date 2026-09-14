/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { modelsCreateDeploymentBodyNameMax } from '@nemo/sdk/generated/platform/zod/model-deployments';
import { WIZARD_DEPLOYMENT_NAME_SUFFIX } from '@studio/routes/NewDeploymentRoute/schema';

/**
 * Hugging Face–source fileset name created in `createHuggingFaceDeployment` inside
 * `useCreateDeploymentBySource`. Keep in sync when changing the wizard.
 */
export function huggingFaceSourceFilesetName(deploymentName: string): string {
  return `${deploymentName}-hf-src`;
}

export const HUGGING_FACE_DEPLOYMENT_SOURCE_FIELD = 'studio_deployment_source';
export const HUGGING_FACE_DEPLOYMENT_SOURCE_VALUE = 'huggingface';

/**
 * Budget for an auto-derived base name.
 *
 * The fileset (`<base>-deployment-hf-src`) is the longest name derived from the
 * wizard base name, so it — not the deployment — sets the ceiling. `schema.ts`
 * validates against the shorter deployment suffix because NGC and Workspace
 * deployments have no fileset; we only hold generated names to the stricter bound.
 */
const HF_DERIVED_SUFFIX = `${WIZARD_DEPLOYMENT_NAME_SUFFIX}-hf-src`;
export const HF_DERIVED_BASE_NAME_MAX_LEN =
  modelsCreateDeploymentBodyNameMax - HF_DERIVED_SUFFIX.length;

/** Anything outside the deployment-name character class. */
const DISALLOWED_CHARS = /[^a-z0-9\-@.+_]+/g;

/**
 * Derive a wizard base name from a Hugging Face repo ID as `<org>-<repo>`.
 *
 *     Qwen/Qwen2.5-7B-Instruct  -> qwen-qwen2.5-7b-instruct
 *     deepseek-ai/DeepSeek-V4   -> deepseek-ai-deepseek-v4
 *     01-ai/Yi-Large            -> m-01-ai-yi-large
 *
 * The result satisfies `modelsCreateDeploymentBodyNameRegExp`: it starts with a
 * lowercase letter, contains no `--`, and does not end with `-`. Dots survive, so
 * version numbers stay readable.
 *
 * Returns `null` when the repo ID yields nothing usable, so callers can leave
 * whatever default is already in the field.
 */
export function huggingFaceRepoIdToBaseName(repoId: string): string | null {
  const trimmed = repoId.trim();
  if (!trimmed) return null;

  // `<org>/<repo>` -> `<org>-<repo>`; a bare `<repo>` keeps its single segment.
  let slug = trimmed
    .split('/')
    .filter(Boolean)
    .join('-')
    .toLowerCase()
    .replace(DISALLOWED_CHARS, '-')
    // The name regex rejects consecutive hyphens anywhere in the string.
    .replace(/-{2,}/g, '-')
    .replace(/^-+/, '');

  // A name must start with a lowercase letter. Repos beginning with a digit
  // (`01-ai/...`) are prefixed rather than truncated, matching the `m-` prefix the
  // platform already applies to autodiscovered entities.
  if (slug && !/^[a-z]/.test(slug)) slug = `m-${slug}`;

  // Truncate before stripping the trailing hyphen: the cut itself can leave one.
  slug = slug.slice(0, HF_DERIVED_BASE_NAME_MAX_LEN).replace(/-+$/, '');

  // The regex requires at least two characters.
  return slug.length >= 2 ? slug : null;
}
