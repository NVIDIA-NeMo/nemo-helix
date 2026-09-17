// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  INFERENCE_PARAMETER_FIELDS,
  type InferenceParameterField,
} from '@nemo/common/src/components/ModelSelectV2/InferenceParameters';
import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import type {
  AnonymizerConfig,
  ModelConfig,
  PreviewRequest,
  Rewrite,
  RunJobRequest,
  SelectedModelsOverrides,
} from '@nemo/sdk/generated/anonymizer/schema';
import {
  activeRolesForStrategy,
  ANNOTATE_DEFAULT_TEMPLATE,
  DETECTION_ROLES,
  DEFAULT_MODEL_MAX_TOKENS,
  DEFAULT_MODEL_TEMPERATURE,
  DEFAULT_MODEL_TIMEOUT_SECONDS,
  DEFAULT_MODEL_TOP_P,
  DEFAULT_PREVIEW_ROWS,
  MAX_PREVIEW_ROWS,
  ENTITY_MODE_CUSTOM,
  HASH_ALGORITHM_DEFAULT,
  HASH_ALGORITHM_VALUES,
  HASH_DEFAULT_DIGEST_LENGTH,
  HASH_DEFAULT_TEMPLATE,
  PRIVACY_GOAL_MODE_CUSTOM,
  PRIVACY_GOAL_MODE_DEFAULT,
  REDACT_DEFAULT_TEMPLATE,
  REPLACE_ROLE,
  REWRITE_DEFAULT_MAX_REPAIR_ROUNDS,
  REWRITE_MIN_MAX_REPAIR_ROUNDS,
  REWRITE_ROLES,
  REWRITE_STRATEGY,
  RISK_TOLERANCE_DEFAULT,
  RISK_TOLERANCE_ORDER,
  SOURCE_TYPE_DATASET,
  STRATEGY_ANNOTATE,
  STRATEGY_HASH,
  STRATEGY_REDACT,
  STRATEGY_SUBSTITUTE,
  supportsSamplingParams,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import { trimToUndefined } from '@studio/util/strings';
import { z } from 'zod';

const roleModelSchema = z.object({
  modelId: z.string(),
  model: z.string(),
  provider: z.string(),
  params: z.record(z.string(), z.unknown()).optional(),
});

export const anonymizerFormSchema = z
  .object({
    name: z.string().optional(),
    sourceType: z.enum(['url', 'dataset']),
    source: z.string().trim().min(1, 'A data source is required'),
    strategy: z.enum(['substitute', 'redact', 'annotate', 'hash', 'rewrite']),
    previewRows: z.number().int().min(1).max(MAX_PREVIEW_ROWS),
    textColumn: z.string().optional(),
    dataSummary: z.string().optional(),
    entityMode: z.enum([ENTITY_MODE_CUSTOM, 'auto']),
    includeDefaultEntities: z.boolean(),
    entityLabels: z.array(z.string()),
    redactTemplate: z.string(),
    redactNormalizeLabel: z.boolean(),
    annotateTemplate: z.string(),
    hashAlgorithm: z.enum(HASH_ALGORITHM_VALUES),
    hashDigestLength: z.number().int().min(6).max(64),
    hashTemplate: z.string(),
    privacyGoalMode: z.enum([PRIVACY_GOAL_MODE_DEFAULT, PRIVACY_GOAL_MODE_CUSTOM]),
    privacyProtect: z.string(),
    privacyPreserve: z.string(),
    rewriteInstructions: z.string(),
    riskTolerance: z.enum(RISK_TOLERANCE_ORDER),
    maxRepairRounds: z.number().int().min(REWRITE_MIN_MAX_REPAIR_ROUNDS),
    strictEntityProtection: z.boolean(),
    roleModels: z.record(z.string(), roleModelSchema),
  })
  .superRefine((data, ctx) => {
    for (const role of activeRolesForStrategy(data.strategy)) {
      const roleModel = data.roleModels[role];
      if (!roleModel?.model || !roleModel?.provider) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['roleModels', role, 'modelId'],
          message: 'Select a model',
        });
      }
    }

    // Without this the request would carry no detect config and the server would fall back to
    // its own defaults — the opposite of the restricted set Custom mode promises.
    if (
      data.entityMode === ENTITY_MODE_CUSTOM &&
      !data.includeDefaultEntities &&
      !data.entityLabels.length
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['entityLabels'],
        message: 'Select at least one entity label, or include the available entities',
      });
    }
  });

export type AnonymizerFormData = z.infer<typeof anonymizerFormSchema>;

export const getAnonymizerFormDefaults = (): AnonymizerFormData => ({
  name: generateDefaultName(),
  sourceType: SOURCE_TYPE_DATASET,
  source: '',
  strategy: STRATEGY_SUBSTITUTE,
  previewRows: DEFAULT_PREVIEW_ROWS,
  textColumn: '',
  dataSummary: '',
  entityMode: ENTITY_MODE_CUSTOM,
  includeDefaultEntities: true,
  entityLabels: [],
  redactTemplate: REDACT_DEFAULT_TEMPLATE,
  redactNormalizeLabel: true,
  annotateTemplate: ANNOTATE_DEFAULT_TEMPLATE,
  hashAlgorithm: HASH_ALGORITHM_DEFAULT,
  hashDigestLength: HASH_DEFAULT_DIGEST_LENGTH,
  hashTemplate: HASH_DEFAULT_TEMPLATE,
  privacyGoalMode: PRIVACY_GOAL_MODE_DEFAULT,
  privacyProtect: '',
  privacyPreserve: '',
  rewriteInstructions: '',
  riskTolerance: RISK_TOLERANCE_DEFAULT,
  maxRepairRounds: REWRITE_DEFAULT_MAX_REPAIR_ROUNDS,
  strictEntityProtection: false,
  roleModels: {},
});

const withTemplate = <T extends object>(base: T, template: string): T => {
  const trimmed = template.trim();
  return trimmed ? { ...base, format_template: trimmed } : base;
};

const buildReplaceConfig = (form: AnonymizerFormData): AnonymizerConfig['replace'] => {
  const replace = ((): object => {
    switch (form.strategy) {
      case STRATEGY_REDACT:
        return withTemplate(
          { kind: STRATEGY_REDACT, normalize_label: form.redactNormalizeLabel },
          form.redactTemplate
        );
      case STRATEGY_ANNOTATE:
        return withTemplate({ kind: STRATEGY_ANNOTATE }, form.annotateTemplate);
      case STRATEGY_HASH:
        return withTemplate(
          {
            kind: STRATEGY_HASH,
            algorithm: form.hashAlgorithm,
            digest_length: form.hashDigestLength,
          },
          form.hashTemplate
        );
      default:
        return { kind: STRATEGY_SUBSTITUTE };
    }
  })();
  return replace as AnonymizerConfig['replace'];
};

const buildRewriteConfig = (form: AnonymizerFormData): Rewrite => {
  const rewrite: Rewrite = {
    risk_tolerance: form.riskTolerance,
    max_repair_iterations: form.maxRepairRounds,
    strict_entity_protection: form.strictEntityProtection,
  };

  const instructions = trimToUndefined(form.rewriteInstructions);
  if (instructions) {
    rewrite.instructions = instructions;
  }
  if (form.privacyGoalMode === PRIVACY_GOAL_MODE_CUSTOM) {
    rewrite.privacy_goal = {
      protect: form.privacyProtect.trim(),
      preserve: form.privacyPreserve.trim(),
    };
  }

  return rewrite;
};

/**
 * entity_labels replaces the default set server-side, so "include defaults" has to send the
 * defaults alongside the custom picks. Omitted entirely when the selection adds nothing, which
 * leaves the server on its own defaults.
 */
const buildDetectConfig = (
  form: AnonymizerFormData,
  defaultEntityLabels: string[]
): AnonymizerConfig['detect'] => {
  if (form.entityMode !== ENTITY_MODE_CUSTOM) return undefined;

  const labels = form.includeDefaultEntities
    ? [...new Set([...defaultEntityLabels, ...form.entityLabels])]
    : form.entityLabels;

  if (!labels.length) return undefined;
  if (form.includeDefaultEntities && labels.length === new Set(defaultEntityLabels).size) {
    return undefined;
  }

  return { entity_labels: labels };
};

/**
 * Sampling params are dropped for the GLiNER detector rather than merely hidden, so a role that
 * once held them (or a strategy switch) can't leave them on a model that ignores them.
 */
const buildRoleParams = (
  role: string,
  overrides: Record<string, unknown> = {}
): Record<string, unknown> => {
  if (!supportsSamplingParams(role)) {
    const kept = Object.entries(overrides).filter(
      ([key]) => !INFERENCE_PARAMETER_FIELDS.includes(key as InferenceParameterField)
    );
    return { timeout: DEFAULT_MODEL_TIMEOUT_SECONDS, ...Object.fromEntries(kept) };
  }

  return {
    timeout: DEFAULT_MODEL_TIMEOUT_SECONDS,
    max_tokens: DEFAULT_MODEL_MAX_TOKENS,
    temperature: DEFAULT_MODEL_TEMPERATURE,
    top_p: DEFAULT_MODEL_TOP_P,
    ...overrides,
  };
};

export const buildAnonymizerJobRequest = (
  form: AnonymizerFormData,
  defaultEntityLabels: string[] = []
): RunJobRequest => {
  const config: AnonymizerConfig =
    form.strategy === REWRITE_STRATEGY
      ? { rewrite: buildRewriteConfig(form) }
      : { replace: buildReplaceConfig(form) };

  const detect = buildDetectConfig(form, defaultEntityLabels);
  if (detect) {
    config.detect = detect;
  }

  const aliasByModel = new Map<string, string>();
  const modelConfigs: ModelConfig[] = [];
  const aliasForRole: Record<string, string> = {};

  for (const role of activeRolesForStrategy(form.strategy)) {
    const roleModel = form.roleModels[role];
    const model = roleModel?.model.trim() ?? '';
    const provider = roleModel?.provider.trim() ?? '';
    const params = buildRoleParams(role, roleModel?.params);
    const key = `${provider}::${model}::${JSON.stringify(params)}`;
    let alias = aliasByModel.get(key);
    if (!alias) {
      alias = `model-${aliasByModel.size + 1}`;
      aliasByModel.set(key, alias);
      modelConfigs.push({
        alias,
        model,
        provider,
        inference_parameters: params as ModelConfig['inference_parameters'],
      });
    }
    aliasForRole[role] = alias;
  }

  const toRoleMap = (roles: string[]) =>
    Object.fromEntries(roles.map((role) => [role, aliasForRole[role]]));

  const selectedModels: SelectedModelsOverrides = { detection: toRoleMap(DETECTION_ROLES) };
  if (form.strategy === REWRITE_STRATEGY) {
    selectedModels.rewrite = toRoleMap(REWRITE_ROLES);
  }
  if (form.strategy === REWRITE_STRATEGY || form.strategy === STRATEGY_SUBSTITUTE) {
    selectedModels.replace = { [REPLACE_ROLE]: aliasForRole[REPLACE_ROLE] };
  }

  return {
    name: trimToUndefined(form.name),
    spec: {
      config,
      data: {
        source: form.source.trim(),
        text_column: trimToUndefined(form.textColumn),
        data_summary: trimToUndefined(form.dataSummary),
      },
      model_configs: modelConfigs,
      selected_models: selectedModels,
    },
  };
};

export const buildAnonymizerPreviewRequest = (
  form: AnonymizerFormData,
  defaultEntityLabels: string[] = []
): PreviewRequest => ({
  ...buildAnonymizerJobRequest(form, defaultEntityLabels).spec,
  num_records: form.previewRows,
});
