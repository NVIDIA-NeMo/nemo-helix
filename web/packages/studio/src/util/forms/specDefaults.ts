// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type AutomodelJobInput,
  type RlJobInput,
  type UnslothJobInput,
} from '@nemo/sdk/generated/customizer/schema';
import { CustomizationCreateAutomodelJobBody } from '@nemo/sdk/generated/customizer/zod/automodel-jobs';
import { CustomizationCreateRlJobBody } from '@nemo/sdk/generated/customizer/zod/rl-jobs';
import { CustomizationCreateUnslothJobBody } from '@nemo/sdk/generated/customizer/zod/unsloth-jobs';

/**
 * Backend defaults for the customizer forms, taken from the Zod schemas Orval generates
 * out of `plugins/nemo-customizer/openapi/openapi.yaml`.
 *
 * Nothing here is hardcoded. The generated schemas carry the spec's `default:` values as
 * `.default(...)`, so parsing a seed that supplies only the required fields makes Zod
 * fill in every default at once — one call per backend instead of a line per control.
 * `pnpm install` re-derives all of them, and a backend default change reaches the form
 * without anyone editing a field list.
 *
 * A field the spec leaves without a `default:` parses to `undefined`, which is the signal
 * for the form to render it unset rather than invent a starting value.
 */

/**
 * Zod applies a nested object's inner defaults only when that object is present in the
 * input, so every container the form binds to has to be seeded with `{}`. Required leaves
 * (`model`, `dataset`) get an empty string: the form overwrites them, and Zod refuses to
 * parse without them.
 */
/** Present so Zod fills the defaults nested inside; without the container it skips them. */
const integrationsSeed = () => ({ wandb: {}, mlflow: {} });

/** Same reason as {@link integrationsSeed}. */
const progressReportingSeed = () => ({ progress_reporting: {} });

export const AUTOMODEL_SEED = {
  model: '',
  dataset: { training: '' },
  training: { lora: {} },
  schedule: progressReportingSeed(),
  batch: {},
  optimizer: {},
  parallelism: {},
  integrations: integrationsSeed(),
};

export const UNSLOTH_SEED = {
  model: { name: '' },
  dataset: { path: '' },
  training: { lora: {} },
  schedule: progressReportingSeed(),
  batch: {},
  optimizer: {},
  hardware: {},
  integrations: integrationsSeed(),
};

/**
 * `training` is a discriminated union, so the seed has to name the arm. The two arms hold
 * genuinely different values — `ref_policy_kl_penalty` is 0.05 on DPO and 0 on GRPO — so
 * each is parsed separately rather than merged.
 */
export const rlSeed = (type: 'dpo' | 'grpo') => ({
  model: '',
  dataset: '',
  training: {
    type,
    parallelism: {},
    ...progressReportingSeed(),
    ...(type === 'grpo' ? { lora: {} } : {}),
  },
  integrations: integrationsSeed(),
});

const specOf = <T>(schema: { parse: (input: unknown) => { spec: T } }, seed: unknown): T =>
  schema.parse({ spec: seed }).spec;

/** Fully-defaulted spec objects. Bind forms to these directly. */
export const AUTOMODEL_DEFAULT_SPEC = specOf<AutomodelJobInput>(
  CustomizationCreateAutomodelJobBody,
  AUTOMODEL_SEED
);
export const UNSLOTH_DEFAULT_SPEC = specOf<UnslothJobInput>(
  CustomizationCreateUnslothJobBody,
  UNSLOTH_SEED
);
export const RL_DPO_DEFAULT_SPEC = specOf<RlJobInput>(CustomizationCreateRlJobBody, rlSeed('dpo'));
export const RL_GRPO_DEFAULT_SPEC = specOf<RlJobInput>(
  CustomizationCreateRlJobBody,
  rlSeed('grpo')
);

/**
 * Flattened `snake_case` view of a parsed spec, e.g. `optimizer_learning_rate`. The slider
 * and placeholder helpers look values up by field path rather than by object traversal, so
 * a control names the field once.
 */
const flatten = (
  value: unknown,
  prefix = '',
  out = new Map<string, unknown>()
): ReadonlyMap<string, unknown> => {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return out;
  for (const [key, child] of Object.entries(value)) {
    const path = prefix ? `${prefix}_${key}` : key;
    out.set(path, child);
    if (child !== null && typeof child === 'object' && !Array.isArray(child)) {
      flatten(child, path, out);
    }
  }
  return out;
};

export const AUTOMODEL_SPEC_DEFAULTS = flatten(AUTOMODEL_DEFAULT_SPEC);
export const UNSLOTH_SPEC_DEFAULTS = flatten(UNSLOTH_DEFAULT_SPEC);
/** Keyed off `training`, so callers use `epochs` / `parallelism_num_nodes`. */
export const DPO_SPEC_DEFAULTS = flatten(RL_DPO_DEFAULT_SPEC.training);
export const GRPO_SPEC_DEFAULTS = flatten(RL_GRPO_DEFAULT_SPEC.training);

/** Reads a spec default, narrowed to the type the caller expects. `undefined` means unset. */
const reader =
  <T>(guard: (value: unknown) => value is T) =>
  (defaults: ReadonlyMap<string, unknown>, field: string): T | undefined => {
    const value = defaults.get(field);
    return guard(value) ? value : undefined;
  };

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const isBoolean = (value: unknown): value is boolean => typeof value === 'boolean';
const isString = (value: unknown): value is string => typeof value === 'string';

export const numberDefault = reader(isNumber);
export const booleanDefault = reader(isBoolean);
export const stringDefault = reader(isString);

/**
 * What a control shows when the spec gives it no default. Kept here so every form uses the
 * same word for "the backend decides this", rather than each control inventing a hint.
 */
export const UNSET_PLACEHOLDER = 'Unset';

/**
 * Placeholder for a slider: the backend's own default when the spec has one, so the user
 * can see what will happen without the form sending a value, and `Unset` when it does not.
 */
export const placeholderFor = (defaults: ReadonlyMap<string, unknown>, field: string): string => {
  const value = defaults.get(field);
  return value === undefined || value === null ? UNSET_PLACEHOLDER : String(value);
};

/**
 * Slider props for a spec-backed field: the backend's default as the seeded value and ↺
 * target, or unset with an `Unset` placeholder when the spec declares no default.
 *
 * Spread rather than passed field-by-field so a control cannot end up seeded from one
 * field and placeholdered from another.
 */
export const specSliderProps = (
  defaults: ReadonlyMap<string, unknown>,
  field: string
): { defaultValue: number | undefined; unsetPlaceholder: string } => ({
  defaultValue: numberDefault(defaults, field),
  unsetPlaceholder: placeholderFor(defaults, field),
});
