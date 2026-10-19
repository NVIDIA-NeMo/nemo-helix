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

import { resourceRefSchema } from '@nemo/common/src/types';
import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import { Engine } from '@nemo/sdk/generated/platform/schema';
import {
  modelsCreateDeploymentBodyNameMax,
  modelsCreateDeploymentBodyNameRegExp,
} from '@nemo/sdk/generated/platform/zod/model-deployments';
import { z } from 'zod';

/** Suffixes appended to the wizard base name for API resource names. */
export const WIZARD_DEPLOYMENT_NAME_SUFFIX = '-deployment' as const;
export const WIZARD_CONFIG_NAME_SUFFIX = '-config' as const;

const baseNameMaxLen = modelsCreateDeploymentBodyNameMax - WIZARD_DEPLOYMENT_NAME_SUFFIX.length;

const wizardBaseNameSchema = z
  .string()
  .trim()
  .min(1, 'Name is required')
  .max(
    baseNameMaxLen,
    `Name must be at most ${baseNameMaxLen} characters so “${WIZARD_DEPLOYMENT_NAME_SUFFIX}” can be appended`
  )
  .regex(
    modelsCreateDeploymentBodyNameRegExp,
    'Allowed characters: letters, digits, underscores, hyphens, and dots'
  );

export function deploymentNameFromWizardBaseName(baseName: string): string {
  return `${baseName}${WIZARD_DEPLOYMENT_NAME_SUFFIX}`;
}

export function configNameFromWizardBaseName(baseName: string): string {
  return `${baseName}${WIZARD_CONFIG_NAME_SUFFIX}`;
}

export const SOURCE_NGC = 'ngc' as const;
export const SOURCE_HF = 'huggingface' as const;
export const SOURCE_WORKSPACE = 'workspace' as const;

export const WORKSPACE_PICKER_MODEL = 'model' as const;
export const WORKSPACE_PICKER_FILESET = 'fileset' as const;

/**
 * Engines that need an explicit image.
 *
 * `vllm` resolves to a model-agnostic default server image, so leaving the image
 * blank is safe. `nim` falls back to a model-specific NIM (Llama-3.1-8B today),
 * which silently mismatches any other architecture, and `generic` has no
 * compiler at all — both need the user to say which image to run.
 */
export const ENGINES_REQUIRING_IMAGE: readonly Engine[] = [Engine.nim, Engine.generic];

export function engineRequiresImage(engine: Engine): boolean {
  return ENGINES_REQUIRING_IMAGE.includes(engine);
}

/** Sources where the user picks the engine; NGC is a NIM container by definition. */
export function sourceSupportsEngineChoice(source: WizardFormValues['source']): boolean {
  return source === SOURCE_HF || source === SOURCE_WORKSPACE;
}

const additionalEnvRowSchema = z.object({
  key: z.string(),
  value: z.string().optional(),
});

/**
 * Require an explicit image for engines whose default would otherwise be wrong.
 *
 * Only applies to the sources that expose an engine picker — the NGC source
 * already requires an image unconditionally.
 */
function requireImageForEngine(
  data: { engine: Engine; imageName?: string; imageTag?: string },
  ctx: z.RefinementCtx
): void {
  if (!engineRequiresImage(data.engine)) return;

  const label = data.engine === Engine.nim ? 'NIM' : 'generic';
  if (!data.imageName?.trim()) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `Image name is required for the ${label} engine`,
      path: ['imageName'],
    });
  }
  if (!data.imageTag?.trim()) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `Image tag is required for the ${label} engine`,
      path: ['imageTag'],
    });
  }
}

export const createDeploymentWizardSchema = z
  .object({
    source: z.enum([SOURCE_HF, SOURCE_WORKSPACE, SOURCE_NGC]),
    /** Base name: NGC NIM `model_name`, and API deployment/config become `<name>-deployment` / `<name>-config`. */
    name: wizardBaseNameSchema,
    /** Inference engine. Ignored for the NGC source, which is always a NIM container. */
    engine: z.nativeEnum(Engine),
    imageName: z.string().optional(),
    imageTag: z.string().optional(),
    gpu: z.coerce.number().int().min(1, 'At least 1 GPU'),
    loraEnabled: z.boolean(),
    diskSize: z.string().optional(),
    /** Key/value rows mapped to `DeploymentParams.additional_envs` (NGC flow only). */
    additionalEnvs: z.array(additionalEnvRowSchema).default([]),
    repoId: z.string().optional(),
    hfTokenSecret: z.string().optional().or(z.literal('')),
    /** Which kind of workspace resource the user is picking. */
    workspacePickerType: z.enum([WORKSPACE_PICKER_MODEL, WORKSPACE_PICKER_FILESET]),
    /** `<workspace>/<name>` reference to an existing model entity. */
    modelRef: resourceRefSchema.optional().or(z.literal('')),
    /** `<workspace>/<name>` reference to an existing fileset. */
    fileset: resourceRefSchema.optional().or(z.literal('')),
  })
  .superRefine((data, ctx) => {
    if (data.source === SOURCE_NGC) {
      if (!data.imageName?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Image name is required',
          path: ['imageName'],
        });
      }
      if (!data.imageTag?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Image tag is required',
          path: ['imageTag'],
        });
      }
    } else if (data.source === SOURCE_HF) {
      if (!data.repoId?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Hugging Face repo ID is required (e.g. Qwen/Qwen2.5-1.5B-Instruct)',
          path: ['repoId'],
        });
      }
      requireImageForEngine(data, ctx);
    } else {
      requireImageForEngine(data, ctx);
      if (data.workspacePickerType === WORKSPACE_PICKER_MODEL) {
        if (!data.modelRef?.trim()) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'Select a model',
            path: ['modelRef'],
          });
        }
      } else if (!data.fileset?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Select a fileset',
          path: ['fileset'],
        });
      }
    }

    const seenKeys = new Set<string>();
    for (let i = 0; i < data.additionalEnvs.length; i++) {
      const k = data.additionalEnvs[i]?.key?.trim();
      if (!k) continue;
      if (seenKeys.has(k)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Duplicate variable name',
          path: ['additionalEnvs', i, 'key'],
        });
      }
      seenKeys.add(k);
    }
  });

export type WizardFormValues = z.infer<typeof createDeploymentWizardSchema>;

export const defaultWizardValues = (): WizardFormValues => ({
  // Matches the leftmost segment in the panel's source control. HuggingFace is
  // also the only default that agrees with `engine` below: the NGC source
  // ignores the engine entirely, so defaulting to it left the two defaults
  // describing different deployments.
  source: SOURCE_HF,
  name: generateDefaultName(),
  // vLLM serves any architecture from a model-agnostic default image, so it is
  // the safe default for the sources that expose the picker. The NGC source
  // overrides this to `nim` when building its request.
  engine: Engine.vllm,
  imageName: '',
  imageTag: '',
  gpu: 1,
  loraEnabled: true,
  diskSize: '50Gi',
  additionalEnvs: [],
  repoId: '',
  hfTokenSecret: '',
  workspacePickerType: WORKSPACE_PICKER_MODEL,
  modelRef: '',
  fileset: '',
});

/** Build API `additional_envs` from form rows; omits blank keys and all-blank rows. */
export function additionalEnvsFormToApi(
  rows: WizardFormValues['additionalEnvs']
): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const key = row.key?.trim();
    if (!key) continue;
    out[key] = row.value?.trim() ?? '';
  }
  return Object.keys(out).length > 0 ? out : undefined;
}
