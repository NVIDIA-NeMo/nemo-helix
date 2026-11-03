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
import { type ModelSpec } from '@nemo/sdk/generated/platform/schema';
import {
  LEARNING_RATE_DEFAULTS,
  type LearningRateDefaultRule,
} from '@nemo/sdk/generated/schema-extensions/learningRateDefaults';
import { type CustomizationBackend } from '@studio/util/customizationBackend';
import { isPlainObject } from '@studio/util/functions';

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

const LEARNING_RATE_GROUP = {
  automodel: 'optimizer',
  unsloth: 'optimizer',
  rl: 'training',
} as const satisfies Record<CustomizationBackend, string>;

export const hasLearningRate = (backend: CustomizationBackend, spec: unknown): boolean => {
  const group = isPlainObject(spec) ? spec[LEARNING_RATE_GROUP[backend]] : undefined;
  return isPlainObject(group) && group.learning_rate != null;
};

/**
 * The backend only resolves a per-selection learning rate when the request omits it; the
 * spec's single `default:` is the LoRA rate and would pin the wrong value for full weights.
 */
export const withoutLearningRate = <T extends object>(
  backend: CustomizationBackend,
  spec: T
): T => {
  const key = LEARNING_RATE_GROUP[backend];
  const group = (spec as Record<string, unknown>)[key];
  if (!isPlainObject(group)) {
    return spec;
  }
  const rest = { ...group };
  delete rest.learning_rate;
  return { ...spec, [key]: rest };
};

const AUTOMODEL_PARSED_SPEC = specOf<AutomodelJobInput>(
  CustomizationCreateAutomodelJobBody,
  AUTOMODEL_SEED
);
const UNSLOTH_PARSED_SPEC = specOf<UnslothJobInput>(
  CustomizationCreateUnslothJobBody,
  UNSLOTH_SEED
);
const RL_DPO_PARSED_SPEC = specOf<RlJobInput>(CustomizationCreateRlJobBody, rlSeed('dpo'));
const RL_GRPO_PARSED_SPEC = specOf<RlJobInput>(CustomizationCreateRlJobBody, rlSeed('grpo'));

/** Fully-defaulted spec objects, minus the learning rate. Bind forms to these directly. */
export const AUTOMODEL_DEFAULT_SPEC = withoutLearningRate('automodel', AUTOMODEL_PARSED_SPEC);
export const UNSLOTH_DEFAULT_SPEC = withoutLearningRate('unsloth', UNSLOTH_PARSED_SPEC);
export const RL_DPO_DEFAULT_SPEC = withoutLearningRate('rl', RL_DPO_PARSED_SPEC);
export const RL_GRPO_DEFAULT_SPEC = withoutLearningRate('rl', RL_GRPO_PARSED_SPEC);

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

/** Flattened from the parsed specs so the learning rate's spec default is kept. */
export const AUTOMODEL_SPEC_DEFAULTS = flatten(AUTOMODEL_PARSED_SPEC);
export const UNSLOTH_SPEC_DEFAULTS = flatten(UNSLOTH_PARSED_SPEC);
/** Keyed off `training`, so callers use `epochs` / `parallelism_num_nodes`. */
export const DPO_SPEC_DEFAULTS = flatten(RL_DPO_PARSED_SPEC.training);
export const GRPO_SPEC_DEFAULTS = flatten(RL_GRPO_PARSED_SPEC.training);

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

export const LEARNING_RATE_UNSET_HINT =
  'Leave it empty to use the backend default for the selected fine-tuning type, shown in the field.';

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

/**
 * The backend's per-selection learning rate, published as `x-learning-rate-defaults` and
 * matched the way `match_learning_rate_default` matches it: the first rule wins.
 */
const matchLearningRateDefault = (
  rules: readonly LearningRateDefaultRule[],
  values: Readonly<Record<string, string | undefined>>
): number | undefined =>
  rules.find((rule) => Object.entries(rule.when).every(([path, value]) => values[path] === value))
    ?.value;

/**
 * Resolves `auto` from the base model's head type, as `AutomodelJobInput.with_resolved_recipe`
 * does. Stays `auto` (SFT) for a causal LM or an unknown head.
 */
export const resolveAutomodelRecipe = (
  recipe: string | undefined,
  modelSpec: Pick<ModelSpec, 'head_type' | 'is_embedding_model'> | undefined
): string | undefined => {
  if (recipe !== 'auto' || !modelSpec) {
    return recipe;
  }
  const head =
    modelSpec.head_type && modelSpec.head_type !== 'unknown'
      ? modelSpec.head_type
      : modelSpec.is_embedding_model
        ? 'embedding'
        : 'unknown';
  if (head === 'embedding') {
    return 'bi_encoder';
  }
  if (head === 'cross_encoder') {
    return 'cross_encoder';
  }
  return recipe;
};

export type LearningRateSelection =
  /** `recipe: null`: `auto` has not resolved yet. */
  | { backend: 'automodel'; finetuningType?: string; recipe?: string | null }
  | { backend: 'unsloth'; finetuningType?: string }
  | { backend: 'rl'; method: 'dpo' | 'grpo'; finetuningType?: string };

/** Pass the recipe through {@link resolveAutomodelRecipe} first; `auto` reads as SFT. */
export const learningRateDefault = (selection: LearningRateSelection): number | undefined => {
  switch (selection.backend) {
    case 'automodel':
      if (selection.recipe === null) {
        return undefined;
      }
      return (
        matchLearningRateDefault(LEARNING_RATE_DEFAULTS.AutomodelJobInput, {
          'training.recipe': selection.recipe,
          'training.finetuning_type': selection.finetuningType,
        }) ?? numberDefault(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_learning_rate')
      );
    case 'unsloth':
      return (
        matchLearningRateDefault(LEARNING_RATE_DEFAULTS.UnslothJobInput, {
          'training.finetuning_type': selection.finetuningType,
        }) ?? numberDefault(UNSLOTH_SPEC_DEFAULTS, 'optimizer_learning_rate')
      );
    case 'rl': {
      if (selection.method === 'dpo') {
        return numberDefault(DPO_SPEC_DEFAULTS, 'learning_rate');
      }
      return (
        matchLearningRateDefault(LEARNING_RATE_DEFAULTS.RlGRPOTraining, {
          finetuning_type: selection.finetuningType,
        }) ?? numberDefault(GRPO_SPEC_DEFAULTS, 'learning_rate')
      );
    }
  }
};

/** Leaves the rate unset for the backend to pick, showing that pick as the placeholder. */
export const learningRateSliderProps = (
  selection: LearningRateSelection
): { defaultValue: undefined; unsetPlaceholder: string; unsetValue: number | undefined } => {
  const rate = learningRateDefault(selection);
  return {
    defaultValue: undefined,
    unsetPlaceholder: rate === undefined ? UNSET_PLACEHOLDER : String(rate),
    unsetValue: rate,
  };
};
