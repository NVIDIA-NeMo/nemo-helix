// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getEntityReference } from '@nemo/common/src/namedEntity';
import { CustomizationCreateAutomodelJobBody } from '@nemo/sdk/generated/customizer/zod/automodel-jobs';
import { CustomizationCreateRlJobBody } from '@nemo/sdk/generated/customizer/zod/rl-jobs';
import { CustomizationCreateUnslothJobBody } from '@nemo/sdk/generated/customizer/zod/unsloth-jobs';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import type { AnnotatedFilesetFile } from '@studio/hooks/useCustomizationDatasetValidation';
import type { GymEnvironmentManifest } from '@studio/hooks/useGymEnvironmentManifest';
import type { CustomizationBackend, CustomizationJob } from '@studio/util/customizationBackend';
import { getFinetuningType } from '@studio/util/customizations';
import {
  CUSTOMIZER_SCHEMA_LABELS,
  type CustomizerSchemaDetection,
  type CustomizerSchemaVariant,
  type TrainingType,
} from '@studio/util/customizerSchema';
import {
  customizationFormSchema,
  formToAutomodelCreate,
  formToRlCreate,
  formToUnslothCreate,
  jobToFormFields,
  resolveTrainingType,
  stripNulls,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { AUTOMODEL_SEED, rlSeed, UNSLOTH_SEED } from '@studio/util/forms/specDefaults';
import { isPlainObject } from '@studio/util/functions';
import { z } from 'zod';

export interface DraftDataset {
  /** The picked fileset, as `workspace/name`. */
  fileset: string;
  /** Null when the first training row matches no format customizer accepts. */
  schema: CustomizerSchemaDetection | null;
  trainingRowCount: number;
  rowCountIsEstimate: boolean;
  hasValidation: boolean;
  /** Field names and types of the first training row — no values. */
  shape: string;
}

interface RowCount {
  trainingRowCount: number;
  rowCountIsEstimate: boolean;
}

/**
 * Rows across the training files. A count taken from a capped preview — fewer bytes read
 * than the file holds — is scaled up by the file's size, assuming the rows read are typical
 * of the rest. A count from the whole file is used as is.
 *
 * Bytes read are measured on decoded text, which only matches the file for valid UTF-8, so
 * the total is never presented as exact when the encoding check failed.
 */
export const estimateTrainingRows = (files: AnnotatedFilesetFile[], validUtf8 = true): RowCount =>
  files.reduce<RowCount>(
    (total, file) => {
      const rows = file.rowCount ?? 0;
      const bytesRead = file.bytesRead ?? file.size;
      const capped = bytesRead > 0 && bytesRead < file.size;
      return {
        trainingRowCount:
          total.trainingRowCount + (capped ? Math.round((rows * file.size) / bytesRead) : rows),
        rowCountIsEstimate: total.rowCountIsEstimate || capped,
      };
    },
    { trainingRowCount: 0, rowCountIsEstimate: !validUtf8 }
  );

/** The picked reward environment, with what its `nemo-environment.yaml` says about itself. */
export interface DraftEnvironment {
  /** The picked fileset, as `workspace/name`. */
  fileset: string;
  manifest: GymEnvironmentManifest | null;
}

/** The user's picks. Fixed: the draft decides how to train them, never which ones. */
export interface DraftInputs {
  model: ModelEntity;
  dataset: DraftDataset;
  /** Optional, and only used by GRPO. */
  environment: DraftEnvironment | null;
}

/** Methods whose data format a detected variant is in. */
export const methodsForVariant = (variant: CustomizerSchemaVariant): TrainingType[] => {
  if (variant === 'grpo-gym') return ['grpo'];
  if (variant.startsWith('dpo-')) return ['dpo'];
  return ['sft', 'distillation'];
};

const draftSchema = z.object({
  backend: z.enum(['automodel', 'unsloth', 'rl']),
  job: z.record(z.unknown()),
  rationale: z.array(z.string()).default([]),
  needs_from_user: z.array(z.string()).default([]),
});

export type CustomizationDraft = z.infer<typeof draftSchema>;
type Job = Record<string, unknown>;

export const ERROR_NOT_JSON = "The model's reply was not valid JSON.";

const formatIssues = (issues: z.ZodIssue[]): string[] =>
  issues.map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`);

const TO_REQUEST = {
  automodel: formToAutomodelCreate,
  unsloth: formToUnslothCreate,
  rl: formToRlCreate,
};

const JOB_BODIES = {
  automodel: CustomizationCreateAutomodelJobBody,
  unsloth: CustomizationCreateUnslothJobBody,
  rl: CustomizationCreateRlJobBody,
};

const record = (value: unknown): Job => (isPlainObject(value) ? value : {});

/**
 * Replaces every reference with the user's picks — the model is told it may write
 * placeholders — and drops a teacher, which the user picks in the form. The validation
 * reference follows the form's own rule: the dataset itself, when it ships a validation split.
 */
const applyInputs = (
  backend: CustomizationBackend,
  job: Job,
  { model, dataset, environment }: DraftInputs
): Job => {
  const modelRef = getEntityReference(model);
  const validationRef = dataset.hasValidation ? dataset.fileset : undefined;
  if (backend === 'automodel') {
    const training = { ...record(job.training) };
    delete training.teacher_model;
    return {
      ...job,
      model: modelRef,
      dataset: { ...record(job.dataset), training: dataset.fileset, validation: validationRef },
      training,
    };
  }
  if (backend === 'unsloth') {
    return {
      ...job,
      model: { ...record(job.model), name: modelRef },
      dataset: { ...record(job.dataset), path: dataset.fileset, validation_path: validationRef },
    };
  }
  return { ...job, model: modelRef, dataset: dataset.fileset, environment: environment?.fileset };
};

/**
 * Zod applies a nested object's defaults only when the object is present, so each of the
 * seed's containers is merged under the job one level deep before parsing.
 */
const withSeed = (seed: Job, job: Job): Job =>
  Object.fromEntries(
    Object.entries({ ...seed, ...job }).map(([key, value]) => [
      key,
      isPlainObject(seed[key]) && isPlainObject(value) ? { ...seed[key], ...value } : value,
    ])
  );

/**
 * Paths the model wrote that the validator dropped. The generated validators strip unknown
 * keys silently, while the backend rejects them (`extra="forbid"`) — so they are reported,
 * as the backend would, instead of vanishing from the job.
 */
const unknownPaths = (written: Job, kept: Job, prefix = ''): string[] =>
  Object.entries(written).flatMap(([key, value]) => {
    if (value === null || value === undefined) return [];
    const path = prefix ? `${prefix}.${key}` : key;
    if (!(key in kept)) return [path];
    return isPlainObject(value) && isPlainObject(kept[key])
      ? unknownPaths(value, kept[key], path)
      : [];
  });

const seeded = (backend: CustomizationBackend, job: Job): Job => {
  if (backend === 'automodel') return withSeed(AUTOMODEL_SEED, job);
  if (backend === 'unsloth') return withSeed(UNSLOTH_SEED, job);
  const type = record(job.training).type;
  return type === 'dpo' || type === 'grpo' ? withSeed(rlSeed(type), job) : job;
};

/** Keys whose values are the user's picks or bookkeeping, not decisions the draft made. */
const NOT_SETTINGS = new Set([
  'model',
  'dataset',
  'environment',
  'output',
  'name',
  'description',
  'integrations',
  'deployment_config',
]);

export interface DraftSetting {
  label: string;
  value: string;
}

const formatValue = (value: unknown): string => {
  if (typeof value === 'boolean') return value ? 'On' : 'Off';
  if (typeof value === 'number') {
    return value !== 0 && Math.abs(value) < 1e-3 ? value.toExponential() : String(value);
  }
  return typeof value === 'string' ? value : JSON.stringify(value);
};

/** Every leaf the model wrote, by dotted path — what it chose, before any default fills in. */
const settingsOf = (job: Job, prefix = ''): DraftSetting[] =>
  Object.entries(job).flatMap(([key, value]) => {
    if (!prefix && NOT_SETTINGS.has(key)) return [];
    if (value === null || value === undefined) return [];
    const path = prefix ? `${prefix}.${key}` : key;
    return isPlainObject(value)
      ? settingsOf(value, path)
      : [{ label: path, value: formatValue(value) }];
  });

export interface DraftSummary {
  method: TrainingType;
  backend: CustomizationBackend;
  /** Empty for DPO, which has no fine-tuning type: it always trains full weights. */
  finetuningType: string;
  outputName: string;
  baseModel: string;
  dataset: string;
  environment: string | null;
  /** Every value the model wrote; anything absent keeps the backend default. */
  settings: DraftSetting[];
  rationale: string[];
  needsFromUser: string[];
}

export type DraftValidation =
  | {
      status: 'valid';
      values: CustomizationFormFields;
      summary: DraftSummary;
      /** The request the form would submit for these values, pretty-printed. */
      config: string;
    }
  | { status: 'invalid'; errors: string[] };

/**
 * The teacher, and a reward environment the user did not pick here, are picked in the form;
 * the draft says so in `needs_from_user`, and the form flags them at submit.
 */
const DEFERRABLE_FIELDS = new Set(['grpo.environmentFileset', 'automodel.training.teacher_model']);

/**
 * Takes the tool call's raw arguments and accepts the job only when the backend's own job
 * validator and the form's schema both pass. Errors are worded for the model, which gets
 * them back to fix the job.
 */
export const validateDraft = (args: string, inputs: DraftInputs): DraftValidation => {
  let raw: unknown;
  try {
    raw = JSON.parse(args);
  } catch {
    return { status: 'invalid', errors: [ERROR_NOT_JSON] };
  }
  const parsedDraft = draftSchema.safeParse(raw);
  if (!parsedDraft.success) {
    return { status: 'invalid', errors: formatIssues(parsedDraft.error.issues) };
  }
  const draft = parsedDraft.data;
  const { backend } = draft;
  // The skill's templates write unset fields as null, which the backend accepts and the
  // generated validators reject.
  const job = applyInputs(backend, stripNulls(draft.job), inputs);

  const parsed = JOB_BODIES[backend].safeParse({ spec: seeded(backend, job) });
  if (!parsed.success) {
    return {
      status: 'invalid',
      errors: formatIssues(parsed.error.issues).map((error) => error.replace(/^spec\./, '')),
    };
  }
  const spec = parsed.data.spec as Job;
  const unknown = unknownPaths(job, spec);
  if (unknown.length > 0) {
    return {
      status: 'invalid',
      errors: unknown.map((path) => `${path}: not a field of the ${backend} job schema`),
    };
  }

  const asJob = { spec } as unknown as CustomizationJob;
  const values = jobToFormFields(asJob);
  const method = resolveTrainingType(
    values.backend,
    values.automodel.training.training_type,
    values.grpo.trainingType
  );
  const variant = inputs.dataset.schema?.variant;
  if (variant && !methodsForVariant(variant).includes(method)) {
    return {
      status: 'invalid',
      errors: [
        `The dataset is in ${CUSTOMIZER_SCHEMA_LABELS[variant]} format, which only works with ${methodsForVariant(variant).join(' or ')}, but the job trains ${method}.`,
      ],
    };
  }

  const output = record(job.output);
  values.outputName =
    (typeof output.name === 'string' && output.name) ||
    (typeof job.name === 'string' && job.name) ||
    values.outputName;
  values.description = typeof output.description === 'string' ? output.description : '';

  const formErrors = formatIssues(
    (customizationFormSchema.safeParse(values).error?.issues ?? []).filter(
      (issue) => !DEFERRABLE_FIELDS.has(issue.path.join('.'))
    )
  );
  if (formErrors.length > 0) return { status: 'invalid', errors: formErrors };

  return {
    status: 'valid',
    values,
    config: JSON.stringify(TO_REQUEST[backend](values), null, 2),
    summary: {
      method,
      backend,
      finetuningType: getFinetuningType(asJob),
      outputName: values.outputName,
      baseModel: getEntityReference(inputs.model),
      dataset: inputs.dataset.fileset,
      environment: method === 'grpo' ? (inputs.environment?.fileset ?? null) : null,
      settings: settingsOf(job),
      rationale: draft.rationale,
      needsFromUser: draft.needs_from_user,
    },
  };
};
