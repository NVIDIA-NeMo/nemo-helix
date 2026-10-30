// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelWorkspaceGroup } from '@nemo/common/src/api/models/useModels';
import type {
  CreateJobRequest as DataDesignerJobRequest,
  ModelConfig,
} from '@nemo/sdk/generated/data-designer/schema';
import {
  buildColumnsFromConfig,
  MODEL_ALIAS_KEY,
  validateColumns,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import {
  buildModelsFromConfig,
  resolveTemplateModel,
  validateModels,
} from '@studio/routes/DataDesignerJobBuildRoute/models';
import type { JobBuilderSeed } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';

export const DEFAULT_GENERATED_NAME = 'untitled-dataset';
export const DEFAULT_GENERATED_ROWS = 100;

export const ERROR_NO_CONFIG = 'The model did not return a job config.';
export const ERROR_NO_COLUMNS = 'The generated config has no columns.';
export const ERROR_NO_LOADABLE_COLUMNS =
  'None of the generated columns map to a column type the builder can edit.';

/**
 * Outcome of checking an LLM-generated job request against the build route's own rules.
 * Only a `valid` result carries a `jobRequest` — that's what the create route hands to the
 * builder, so an invalid draft can never be loaded.
 */
export type GeneratedConfigValidation =
  | {
      status: 'valid';
      /** Normalized request: model configs resolved to real workspace models. */
      jobRequest: DataDesignerJobRequest;
      seed: JobBuilderSeed;
      /** Non-blocking notes (substituted models, skipped columns). */
      warnings: string[];
    }
  | { status: 'invalid'; errors: string[]; warnings: string[] };

/** Router state key used to hand a generated job request to the build route. */
export interface DataDesignerGeneratedState {
  generatedJobRequest: DataDesignerJobRequest;
}

/**
 * Narrow an unknown router location state to a generated job request, if present.
 * Mirrors `getCloneJobRequestFromState` — the build route treats both the same way.
 */
export const getGeneratedJobRequestFromState = (state: unknown): DataDesignerJobRequest | null => {
  if (!state || typeof state !== 'object' || !('generatedJobRequest' in state)) return null;
  const request = (state as { generatedJobRequest: unknown }).generatedJobRequest;
  if (!request || typeof request !== 'object' || !('spec' in request)) return null;
  return request as DataDesignerJobRequest;
};

/**
 * Turns a job request into builder state — the same conversion the clone path uses, so a
 * generated config lands on the canvas fully editable rather than as opaque JSON.
 */
export const seedFromJobRequest = (jobRequest: DataDesignerJobRequest): JobBuilderSeed => ({
  name: jobRequest.name?.trim() || DEFAULT_GENERATED_NAME,
  rows: String(jobRequest.spec?.num_records ?? DEFAULT_GENERATED_ROWS),
  columns: buildColumnsFromConfig(jobRequest.spec.config),
  models: buildModelsFromConfig(jobRequest.spec.config.model_configs),
  description: jobRequest.description,
  customFields: jobRequest.custom_fields,
});

/** The model that drafted the config, carried through so the columns generate with it too. */
export interface GenerationModel {
  /** Model URN. */
  model: string;
  /** Resource ref of the model's provider; required by Data Designer on every model config. */
  provider: string;
}

/**
 * Points each generated model config at a model that actually exists in the workspace.
 *
 * The LLM writes plausible-looking identifiers (`gpt-4o`, `meta/llama-3`) for models the
 * workspace has never heard of. When the panel knows which model drafted the config, every alias
 * is pointed at that one — it is known to exist, the user picked it, and it makes the draft
 * predictable no matter what the LLM invented. Without that (a bare config, e.g. from a test),
 * we fall back to matching the platform list the way templates do.
 *
 * Left untouched while the model list is still empty and no generation model is known.
 */
const normalizeModelConfigs = (
  configs: ModelConfig[],
  modelGroups: ModelWorkspaceGroup[],
  warnings: string[],
  generationModel?: GenerationModel
): ModelConfig[] => {
  if (generationModel?.model) {
    return configs.map((config) => ({
      ...config,
      model: generationModel.model,
      provider: generationModel.provider || config.provider,
    }));
  }

  if (modelGroups.length === 0) return configs;
  return configs.map((config) => {
    const resolved = resolveTemplateModel(modelGroups, config.model || undefined);
    if (!resolved) return config;
    if (config.model && config.model !== resolved.model) {
      warnings.push(
        `"${config.model}" is not available in this workspace — alias "${config.alias}" now uses "${resolved.model}".`
      );
    }
    return { ...config, model: resolved.model, provider: resolved.provider };
  });
};

/** Column names the builder dropped, e.g. an image column with no palette equivalent. */
const skippedColumnLabels = (
  jobRequest: DataDesignerJobRequest,
  seed: JobBuilderSeed
): string[] => {
  const loaded = new Set(seed.columns.map((column) => column.name));
  return jobRequest.spec.config.columns
    .filter((column) => column.column_type !== 'seed-dataset' && !loaded.has(column.name))
    .map((column) => `${column.name} (${column.column_type})`);
};

/** LLM columns pointing at a `model_alias` that no model config defines. */
const danglingAliasErrors = (seed: JobBuilderSeed): string[] => {
  const aliases = new Set(seed.models.map((model) => model.alias.trim()));
  return seed.columns.flatMap((column) => {
    const alias = column.values[MODEL_ALIAS_KEY]?.trim();
    if (!alias || aliases.has(alias)) return [];
    return [`${column.name}: no model is configured with alias "${alias}".`];
  });
};

/**
 * Decides whether an LLM-generated job request can be loaded into the build route, by running
 * it through the exact conversion and validation the builder itself uses. Anything that would
 * leave the canvas in a broken state is an error and blocks Continue; lossy-but-recoverable
 * adjustments (a substituted model, a skipped column) are warnings.
 */
export const validateGeneratedJobRequest = (
  jobRequest: DataDesignerJobRequest,
  modelGroups: ModelWorkspaceGroup[],
  generationModel?: GenerationModel
): GeneratedConfigValidation => {
  const warnings: string[] = [];
  const config = jobRequest.spec?.config;
  if (!config) return { status: 'invalid', errors: [ERROR_NO_CONFIG], warnings };
  if (!config.columns?.length) return { status: 'invalid', errors: [ERROR_NO_COLUMNS], warnings };

  const numRecords = jobRequest.spec.num_records;
  const rows =
    Number.isInteger(numRecords) && numRecords >= 1 ? numRecords : DEFAULT_GENERATED_ROWS;
  if (rows !== numRecords) {
    warnings.push(`Record count was not a positive whole number — defaulted to ${rows}.`);
  }

  const normalized: DataDesignerJobRequest = {
    ...jobRequest,
    spec: {
      ...jobRequest.spec,
      num_records: rows,
      config: {
        ...config,
        model_configs: normalizeModelConfigs(
          config.model_configs ?? [],
          modelGroups,
          warnings,
          generationModel
        ),
      },
    },
  };

  const seed = seedFromJobRequest(normalized);

  const skipped = skippedColumnLabels(normalized, seed);
  if (skipped.length > 0) {
    warnings.push(
      `Skipped ${skipped.length} column(s) the builder can't edit: ${skipped.join(', ')}.`
    );
  }

  const errors =
    seed.columns.length === 0
      ? [ERROR_NO_LOADABLE_COLUMNS]
      : [...validateColumns(seed.columns), ...danglingAliasErrors(seed)];
  errors.push(...validateModels(seed.models));

  if (errors.length > 0) return { status: 'invalid', errors, warnings };
  return { status: 'valid', jobRequest: normalized, seed, warnings };
};
