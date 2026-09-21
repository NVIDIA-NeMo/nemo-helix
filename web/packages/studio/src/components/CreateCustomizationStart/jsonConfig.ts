// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CustomizationBackend, type CustomizationJob } from '@studio/util/customizationBackend';
import {
  customizationFormSchema,
  jobToFormFields,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import {
  AUTOMODEL_DEFAULT_SPEC,
  RL_DPO_DEFAULT_SPEC,
  RL_GRPO_DEFAULT_SPEC,
  UNSLOTH_DEFAULT_SPEC,
} from '@studio/util/forms/specDefaults';

export type ParseCustomizationJsonResult =
  | { ok: true; fields: CustomizationFormFields; backend: CustomizationBackend }
  | { ok: false; error: string };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * The job payload the API takes is `{ "spec": { … } }`, but the spec on its own is just
 * as recognisable and is what a user tends to have on the clipboard. Accept either, and
 * carry `name`/`description` through when the wrapper is present.
 */
const unwrap = (
  parsed: Record<string, unknown>
): { spec: Record<string, unknown>; name?: string; description?: string } => {
  if (isRecord(parsed.spec)) {
    return {
      spec: parsed.spec,
      name: typeof parsed.name === 'string' ? parsed.name : undefined,
      description: typeof parsed.description === 'string' ? parsed.description : undefined,
    };
  }
  return { spec: parsed };
};

/**
 * Which backend a spec belongs to.
 *
 * `getCustomizationBackend` is deliberately not reused here: its guards key off fields a
 * *server-returned* spec always carries (`parallelism`, `hardware`), and a hand-written
 * config routinely omits them. So check the discriminator first, then those strong
 * markers, then fall back to the shape of `model`/`dataset` — which differ per backend
 * and are the two fields nobody leaves out.
 */
const detectBackend = (spec: Record<string, unknown>): CustomizationBackend | undefined => {
  const training = isRecord(spec.training) ? spec.training : undefined;
  if (training && (training.type === 'dpo' || training.type === 'grpo')) {
    return CustomizationBackend.rl;
  }
  if ('parallelism' in spec) return CustomizationBackend.automodel;
  if ('hardware' in spec) return CustomizationBackend.unsloth;
  // `model` is a bare string for automodel and RL, an object for unsloth.
  if (isRecord(spec.model)) return CustomizationBackend.unsloth;
  // `dataset` is an object for automodel, a string for RL.
  if (isRecord(spec.dataset)) return CustomizationBackend.automodel;
  if (typeof spec.dataset === 'string' && training) return CustomizationBackend.rl;
  return undefined;
};

/**
 * Overlays `source` onto `base` recursively so a partial config inherits every default it
 * does not mention. Arrays and nulls replace wholesale rather than merging element-wise —
 * a config listing `exclude_modules` means that list, not that list appended to ours.
 */
const deepMerge = (base: unknown, source: unknown): unknown => {
  if (!isRecord(base) || !isRecord(source)) return source;
  const merged: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(source)) {
    merged[key] = key in base ? deepMerge(base[key], value) : value;
  }
  return merged;
};

const defaultSpecFor = (backend: CustomizationBackend, spec: Record<string, unknown>): unknown => {
  if (backend === CustomizationBackend.automodel) return AUTOMODEL_DEFAULT_SPEC;
  if (backend === CustomizationBackend.unsloth) return UNSLOTH_DEFAULT_SPEC;
  const training = isRecord(spec.training) ? spec.training : undefined;
  return training?.type === 'grpo' ? RL_GRPO_DEFAULT_SPEC : RL_DPO_DEFAULT_SPEC;
};

/** The first few zod issues, as `path: message`, so the panel can say what to fix. */
const formatIssues = (issues: { path: PropertyKey[]; message: string }[]): string =>
  issues
    .slice(0, 3)
    .map((issue) => {
      const path = issue.path.map(String).join('.');
      return path ? `${path}: ${issue.message}` : issue.message;
    })
    .join('; ');

/**
 * Turns a pasted or uploaded job config into form values, or explains why it can't.
 *
 * The result is always a complete, schema-valid set of fields: the config is merged over
 * the backend's defaults before validation, so a config naming only the handful of
 * settings a user cares about still opens a fully-populated form.
 */
export const parseCustomizationJson = (text: string): ParseCustomizationJsonResult => {
  const trimmed = text.trim();
  if (!trimmed) return { ok: false, error: 'Paste a job config, or upload a .json file.' };

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (e) {
    return { ok: false, error: `Not valid JSON — ${(e as Error).message}` };
  }

  if (!isRecord(parsed)) {
    return { ok: false, error: 'Expected a JSON object describing a job, not an array or value.' };
  }

  const { spec, name, description } = unwrap(parsed);
  const backend = detectBackend(spec);
  if (!backend) {
    return {
      ok: false,
      error:
        'Could not tell which backend this config is for. Expected an Automodel, Unsloth or RL job spec — check that "model" and "dataset" are present.',
    };
  }

  const mergedSpec = deepMerge(defaultSpecFor(backend, spec), spec);

  // `jobToFormFields` is the same mapping a cloned job goes through, so a config loaded
  // here lands in exactly the state cloning that job would produce.
  const base = jobToFormFields({
    name: name ?? '',
    description,
    spec: mergedSpec,
  } as CustomizationJob);
  // `jobToFormFields` always mints a fresh output name, which is right for cloning a job
  // but not for a config that names the model it means to produce.
  const fields: CustomizationFormFields = name ? { ...base, outputName: name } : base;

  const validation = customizationFormSchema.safeParse(fields);
  if (!validation.success) {
    return { ok: false, error: `Config is not valid — ${formatIssues(validation.error.issues)}` };
  }

  return { ok: true, fields, backend };
};

/**
 * The inverse of {@link parseCustomizationJson}: what a job details page hands back so the
 * same job can be started again. Only the submitter's own fields — the server's id, status
 * and timestamps describe a run that already happened and mean nothing to a new one.
 */
export const customizationJobToConfigJson = (job: CustomizationJob): string =>
  `${JSON.stringify(
    {
      name: job.name,
      ...(job.description ? { description: job.description } : {}),
      spec: job.spec,
    },
    null,
    2
  )}\n`;
