// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { generateDefaultName } from '@nemo/common/src/utils/generateDefaultName';
import {
  type AutomodelJobInput,
  type AutomodelJobsJobRequest,
  RlGRPOTrainingFinetuningType,
  type RlDPOTraining,
  type RlGRPOTraining,
  type RlJobInput,
  type RlJobsJobRequest,
  type UnslothJobInput,
  type IntegrationsSpec,
  type UnslothJobsJobRequest,
} from '@nemo/sdk/generated/customizer/schema';
import { CustomizationCreateAutomodelJobBody } from '@nemo/sdk/generated/customizer/zod/automodel-jobs';
import { CustomizationCreateRlJobBody } from '@nemo/sdk/generated/customizer/zod/rl-jobs';
import { CustomizationCreateUnslothJobBody } from '@nemo/sdk/generated/customizer/zod/unsloth-jobs';
import {
  CustomizationBackend,
  isAutomodelJob,
  isCustomizationBackend,
  isRlJob,
  isAutomodelSpec,
  isRlSpec,
  isUnslothSpec,
  type CustomizationJob,
} from '@studio/util/customizationBackend';
import type { TrainingType } from '@studio/util/customizerSchema';
import {
  AUTOMODEL_DEFAULT_SPEC,
  RL_DPO_DEFAULT_SPEC,
  RL_GRPO_DEFAULT_SPEC,
  UNSLOTH_DEFAULT_SPEC,
} from '@studio/util/forms/specDefaults';
import { z } from 'zod';

/**
 * GRPO-only hyperparameters, kept in their own namespace because `rl.training` holds
 * a single object shared with the DPO form. Derived from `RlGRPOTraining` so the form
 * cannot drift from the API — field docs come from the generated schema.
 */
/**
 * GRPO-only form state: the spec's GRPO arm plus the two fields that exist only in the UI.
 *
 * A field the form has no control for is inert — `formToRlCreate` picks what it sends.
 */
export interface GrpoFormFields extends RlGRPOTraining {
  /** 'grpo' shows the GRPO form sections; 'dpo' shows DPO sections. Maps to training.type on submit. */
  trainingType: RlDPOTraining['type'] | RlGRPOTraining['type'];
  /** Fileset reference for the NeMo Gym reward environment. */
  environmentFileset: string;
}

export interface CustomizationFormFields {
  backend: CustomizationBackend;
  outputName: string;
  description: string;
  automodel: AutomodelJobInput;
  unsloth: UnslothJobInput;
  rl: RlJobInput;
  grpo: GrpoFormFields;
}

type DatasetFieldName = 'automodel.dataset.training' | 'unsloth.dataset.path' | 'rl.dataset';

/** Each backend keeps the training dataset reference under a different path. */
export const DATASET_FIELD_BY_BACKEND: Record<CustomizationBackend, DatasetFieldName> = {
  automodel: 'automodel.dataset.training',
  unsloth: 'unsloth.dataset.path',
  rl: 'rl.dataset',
};

type ModelFieldName = 'automodel.model' | 'unsloth.model.name' | 'rl.model';

/** Likewise for the base model reference. */
export const MODEL_FIELD_BY_BACKEND: Record<CustomizationBackend, ModelFieldName> = {
  automodel: 'automodel.model',
  unsloth: 'unsloth.model.name',
  rl: 'rl.model',
};

/** Unsloth is SFT-only; automodel and RL each carry their own method selector. */
export const resolveTrainingType = (
  backend: CustomizationBackend,
  automodelTrainingType: CustomizationFormFields['automodel']['training']['training_type'],
  rlTrainingType: GrpoFormFields['trainingType'] | undefined
): TrainingType => {
  switch (backend) {
    case 'rl':
      return rlTrainingType ?? 'dpo';
    case 'automodel':
      return automodelTrainingType ?? 'sft';
    default:
      return 'sft';
  }
};

/**
 * Every default below comes from `specDefaults.ts`, which parses each backend's generated
 * Zod schema once and lets Zod apply the spec's `default:` values. A field the spec leaves
 * without a default arrives `undefined` — the form's signal to render it unset rather than
 * invent a starting value.
 */

/**
 * `rl.training` is shared by the DPO and GRPO forms, so each method's defaults must stand
 * alone: a GRPO value left in the DPO object would silently override the backend. Each
 * comes from its own arm of the discriminated union, so values that differ per method —
 * `ref_policy_kl_penalty` is 0.05 on DPO and 0 on GRPO — stay correct without restating.
 */
export const RL_DPO_TRAINING_DEFAULTS = RL_DPO_DEFAULT_SPEC.training;

const grpoTraining = RL_GRPO_DEFAULT_SPEC.training as RlGRPOTraining;

/**
 * Both arms come straight from the spec. The GRPO arm carries its own values for anything
 * the two methods differ on, so nothing needs restating here — and a field the spec leaves
 * without a default stays absent, which is how it reaches the form unset.
 */
export const RL_GRPO_TRAINING_DEFAULTS = RL_GRPO_DEFAULT_SPEC.training;

export const FORM_DEFAULTS: CustomizationFormFields = {
  backend: 'automodel',
  outputName: '',
  description: '',
  automodel: AUTOMODEL_DEFAULT_SPEC,
  unsloth: UNSLOTH_DEFAULT_SPEC,
  rl: RL_DPO_DEFAULT_SPEC,
  grpo: {
    /** 'grpo' shows the GRPO sections; 'dpo' shows DPO. Maps to training.type on submit. */
    trainingType: 'dpo',
    environmentFileset: '',
    ...grpoTraining,
  },
};

const automodelBodySpec = CustomizationCreateAutomodelJobBody.shape.spec;
const automodelSpecSchema = automodelBodySpec
  .omit({ output: true })
  .extend({
    model: z.string().min(1, 'Please select a model'),
    dataset: automodelBodySpec.shape.dataset.extend({
      training: z.string().min(1, 'Training dataset is required'),
    }),
  })
  .superRefine((spec, ctx) => {
    if (spec.training.training_type === 'distillation' && !spec.training.teacher_model) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Teacher model is required for distillation',
        path: ['training', 'teacher_model'],
      });
    }
  });

const unslothBodySpec = CustomizationCreateUnslothJobBody.shape.spec;
const unslothSpecSchema = unslothBodySpec.omit({ output: true }).extend({
  model: unslothBodySpec.shape.model.extend({
    name: z.string().min(1, 'Please select a model'),
  }),
  dataset: unslothBodySpec.shape.dataset.extend({
    path: z.string().min(1, 'Training dataset is required'),
  }),
});

const rlBodySpec = CustomizationCreateRlJobBody.shape.spec;
const rlSpecSchema = rlBodySpec.omit({ output: true }).extend({
  model: z.string().min(1, 'Please select a model'),
  dataset: z.string().min(1, 'Training dataset is required'),
});

export const customizationFormSchema = z
  .object({
    backend: z.nativeEnum(CustomizationBackend),
    outputName: z.string().min(1, 'Output model name is required'),
    description: z.string(),
    automodel: z.unknown(),
    unsloth: z.unknown(),
    rl: z.unknown(),
    grpo: z.unknown(),
  })
  .superRefine((data, ctx) => {
    let spec: z.ZodTypeAny;
    let value: unknown;
    if (data.backend === 'automodel') {
      spec = automodelSpecSchema;
      value = data.automodel;
    } else if (data.backend === 'unsloth') {
      spec = unslothSpecSchema;
      value = data.unsloth;
    } else {
      spec = rlSpecSchema;
      value = data.rl;
      const grpo = data.grpo as Partial<GrpoFormFields> | undefined;
      if (grpo?.trainingType === 'grpo') {
        if (!grpo.environmentFileset) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: 'A reward environment fileset is required for GRPO training',
            path: ['grpo', 'environmentFileset'],
          });
        }
        const training = (data.rl as RlJobInput | undefined)?.training;
        // Mirrors the backend's _generation_length_fits_context validator: max_seq_length
        // is the whole prompt + generation budget, so a larger generation cap is
        // unsatisfiable and the job is rejected at submit.
        const maxSeqLength = training?.max_seq_length;
        if (
          grpo.max_new_tokens != null &&
          maxSeqLength != null &&
          grpo.max_new_tokens > maxSeqLength
        ) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: `Max new tokens cannot exceed the max sequence length (${maxSeqLength}), which is the total prompt + generation budget`,
            path: ['grpo', 'max_new_tokens'],
          });
        }
        // Mirrors the backend's validate_for_training: a remainder otherwise surfaces
        // only as an assert at the first optimizer step, once the job holds GPUs.
        const batchSize = training?.batch_size;
        const generations = grpo.num_generations_per_prompt;
        // Omitting num_prompts_per_step does not skip the rule: the backend derives it,
        // and the floor division is the case its own comment flags as the failure.
        const prompts =
          grpo.num_prompts_per_step ??
          (batchSize != null && generations
            ? Math.max(Math.floor(batchSize / generations), 1)
            : null);
        const rolloutBatch = prompts != null && generations != null ? prompts * generations : null;
        if (
          rolloutBatch != null &&
          batchSize != null &&
          batchSize > 0 &&
          rolloutBatch % batchSize !== 0
        ) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: `Prompts per step × rollouts per prompt (${rolloutBatch}) must be a multiple of the global batch size (${batchSize})`,
            path: ['grpo', 'num_prompts_per_step'],
          });
        }
      }
    }
    const result = spec.safeParse(value);
    if (!result.success) {
      for (const issue of result.error.issues) {
        ctx.addIssue({ ...issue, path: [data.backend, ...issue.path] });
      }
    }
  });

export const formToAutomodelCreate = (f: CustomizationFormFields): AutomodelJobsJobRequest => {
  const { training } = f.automodel;
  const usesLora =
    training.finetuning_type === 'lora' || training.finetuning_type === 'lora_merged';
  const isDistillation = training.training_type === 'distillation';
  return {
    name: f.outputName || undefined,
    description: f.description || undefined,
    spec: {
      ...f.automodel,
      integrations: cleanIntegrations(f.automodel.integrations),
      training: {
        ...training,
        lora: usesLora ? training.lora : undefined,
        teacher_model: isDistillation ? training.teacher_model || undefined : undefined,
        teacher_precision: isDistillation ? training.teacher_precision : undefined,
        distillation_ratio: isDistillation ? training.distillation_ratio : undefined,
        distillation_temperature: isDistillation ? training.distillation_temperature : undefined,
        offload_teacher: isDistillation ? training.offload_teacher : undefined,
      },
      output: { name: f.outputName, description: f.description || undefined },
    },
  };
};

/**
 * Blank text fields mean "no integration", not "an integration named empty string". Drop
 * empty values, then drop a provider whose fields are all empty, then the whole block.
 */
const cleanIntegrations = (
  integrations: IntegrationsSpec | undefined | null
): IntegrationsSpec | undefined => {
  if (!integrations) return undefined;
  const prune = <T extends object>(obj: T | undefined | null): T | undefined => {
    if (!obj) return undefined;
    const kept = Object.entries(obj).filter(([, v]) =>
      typeof v === 'string' ? v.trim() !== '' : Array.isArray(v) ? v.length > 0 : v != null
    );
    return kept.length > 0 ? (Object.fromEntries(kept) as T) : undefined;
  };
  const wandb = prune(integrations.wandb);
  const mlflow = prune(integrations.mlflow);
  return wandb || mlflow ? { ...(wandb && { wandb }), ...(mlflow && { mlflow }) } : undefined;
};

/**
 * react-hook-form materialises a parent object as soon as one of its nested controls
 * registers, so an untouched group arrives as `{}` (or all-undefined) rather than absent.
 * Sending that is not the same as sending nothing: the backend rejects a reward_shaping
 * with no penalty set, and would read an empty reward_scaling as an identity mapping the
 * user never asked for.
 */
const omitIfEmpty = <T extends object>(group: T | undefined | null): T | undefined => {
  if (!group) return undefined;
  const kept = Object.entries(group).filter(([, v]) => v != null);
  return kept.length > 0 ? (Object.fromEntries(kept) as T) : undefined;
};

/** An empty list is not a filter — send nothing so the backend keeps its own default. */
const emptyToUndefined = (v: string[] | undefined | null) => (v && v.length > 0 ? v : undefined);

export const formToRlCreate = (f: CustomizationFormFields): RlJobsJobRequest => {
  if (f.grpo.trainingType === 'grpo') {
    const t = f.rl.training;
    const isLora = f.grpo.finetuning_type === RlGRPOTrainingFinetuningType.lora;
    return {
      name: f.outputName || undefined,
      description: f.description || undefined,
      spec: {
        model: f.rl.model,
        dataset: f.rl.dataset,
        environment: f.grpo.environmentFileset || undefined,
        integrations: cleanIntegrations(f.rl.integrations),
        training: {
          type: 'grpo',
          optimizer_type: t.optimizer_type,
          learning_rate: t.learning_rate,
          min_learning_rate: t.min_learning_rate,
          weight_decay: t.weight_decay,
          adam_beta1: t.adam_beta1,
          adam_beta2: t.adam_beta2,
          adam_eps: t.adam_eps,
          warmup_steps: t.warmup_steps,
          max_steps: t.max_steps,
          val_check_interval: t.val_check_interval,
          val_at_end: t.val_at_end,
          keep_top_k: t.keep_top_k,
          batch_size: t.batch_size,
          micro_batch_size: t.micro_batch_size,
          activation_checkpointing: t.activation_checkpointing,
          max_seq_length: t.max_seq_length,
          seed: t.seed,
          parallelism: t.parallelism,
          max_grad_norm: t.max_grad_norm,
          finetuning_type: f.grpo.finetuning_type,
          lora: isLora
            ? {
                ...f.grpo.lora,
                target_modules: emptyToUndefined(f.grpo.lora?.target_modules),
                exclude_modules: emptyToUndefined(f.grpo.lora?.exclude_modules),
              }
            : undefined,
          num_generations_per_prompt: f.grpo.num_generations_per_prompt,
          num_prompts_per_step: f.grpo.num_prompts_per_step,
          overlong_filtering: f.grpo.overlong_filtering,
          temperature: f.grpo.temperature,
          max_new_tokens: f.grpo.max_new_tokens,
          val_at_start: f.grpo.val_at_start,
          normalize_rewards: f.grpo.normalize_rewards,
          max_rollout_turns: f.grpo.max_rollout_turns,
          ref_policy_kl_penalty: t.ref_policy_kl_penalty,
          ratio_clip_min: f.grpo.ratio_clip_min,
          ratio_clip_max: f.grpo.ratio_clip_max,
          epochs: t.epochs,
          execution_profile: t.execution_profile || undefined,
          ratio_clip_c: f.grpo.ratio_clip_c,
          advantage_clip_low: f.grpo.advantage_clip_low,
          advantage_clip_high: f.grpo.advantage_clip_high,
          use_dynamic_sampling: f.grpo.use_dynamic_sampling,
          // The backend rejects a multiplier other than 1.0 unless dynamic sampling is on,
          // so the two always travel together.
          batch_multiplier: f.grpo.use_dynamic_sampling ? f.grpo.batch_multiplier : undefined,
          dynamic_sampling_max_gen_batches: f.grpo.use_dynamic_sampling
            ? f.grpo.dynamic_sampling_max_gen_batches
            : undefined,
          use_leave_one_out_baseline: f.grpo.use_leave_one_out_baseline,
          use_importance_sampling_correction: f.grpo.use_importance_sampling_correction,
          use_on_policy_kl_approximation: f.grpo.use_on_policy_kl_approximation,
          truncated_importance_sampling_type: f.grpo.truncated_importance_sampling_type,
          truncated_importance_sampling_ratio: f.grpo.truncated_importance_sampling_ratio,
          truncated_importance_sampling_ratio_min: f.grpo.truncated_importance_sampling_ratio_min,
          reward_scaling: omitIfEmpty(f.grpo.reward_scaling),
          reward_shaping: omitIfEmpty(f.grpo.reward_shaping),
          policy_backend: f.grpo.policy_backend,
          batching_strategy: f.grpo.batching_strategy,
          sequence_length_round: f.grpo.sequence_length_round,
          top_k: f.grpo.top_k,
          train_mb_tokens: f.grpo.train_mb_tokens,
          router_aux_loss_coef: f.grpo.router_aux_loss_coef,
          vllm_tensor_parallel_size: f.grpo.vllm_tensor_parallel_size,
          vllm_gpu_memory_utilization: f.grpo.vllm_gpu_memory_utilization,
          hf_config_overrides: f.grpo.hf_config_overrides,
          automodel_kwargs: f.grpo.automodel_kwargs,
        },
        output: { name: f.outputName || undefined },
      },
    };
  }

  // Explicitly pick the DPO-form-exposed fields rather than spreading, so a field the
  // GRPO form binds to can never leak into a DPO request. Every control rendered by
  // GeneralParametersSection's `rl` branch and DpoParametersSection must appear here —
  // anything omitted is a control the user can change that the request then ignores.
  const dpo = f.rl.training as RlDPOTraining;
  return {
    name: f.outputName || undefined,
    description: f.description || undefined,
    spec: {
      model: f.rl.model,
      dataset: f.rl.dataset,
      integrations: cleanIntegrations(f.rl.integrations),
      training: {
        type: 'dpo' as const,
        optimizer_type: dpo.optimizer_type,
        learning_rate: dpo.learning_rate,
        min_learning_rate: dpo.min_learning_rate,
        weight_decay: dpo.weight_decay,
        adam_beta1: dpo.adam_beta1,
        adam_beta2: dpo.adam_beta2,
        adam_eps: dpo.adam_eps,
        warmup_steps: dpo.warmup_steps,
        epochs: dpo.epochs,
        max_steps: dpo.max_steps,
        val_check_interval: dpo.val_check_interval,
        val_at_end: dpo.val_at_end,
        keep_top_k: dpo.keep_top_k,
        batch_size: dpo.batch_size,
        micro_batch_size: dpo.micro_batch_size,
        activation_checkpointing: dpo.activation_checkpointing,
        max_seq_length: dpo.max_seq_length,
        seed: dpo.seed,
        parallelism: dpo.parallelism,
        max_grad_norm: dpo.max_grad_norm,
        ref_policy_kl_penalty: dpo.ref_policy_kl_penalty,
        preference_loss_weight: dpo.preference_loss_weight,
        sft_loss_weight: dpo.sft_loss_weight,
        preference_average_log_probs: dpo.preference_average_log_probs,
        sft_average_log_probs: dpo.sft_average_log_probs,
        execution_profile: dpo.execution_profile || undefined,
      },
      output: { name: f.outputName || undefined },
    },
  };
};

export const formToUnslothCreate = (f: CustomizationFormFields): UnslothJobsJobRequest => {
  const { training } = f.unsloth;
  const usesLora = training?.finetuning_type === 'lora';
  return {
    name: f.outputName || undefined,
    description: f.description || undefined,
    spec: {
      ...f.unsloth,
      integrations: cleanIntegrations(f.unsloth.integrations),
      model: usesLora
        ? f.unsloth.model
        : { ...f.unsloth.model, load_in_4bit: false, load_in_8bit: false },
      hardware: { ...f.unsloth.hardware, gpus: f.unsloth.hardware?.gpus || undefined },
      training: training && { ...training, lora: usesLora ? training.lora : undefined },
      output: { name: f.outputName || undefined, description: f.description || undefined },
    },
  };
};

const stripNulls = <T>(value: T): T => {
  if (value === null) return undefined as unknown as T;
  if (Array.isArray(value)) return value.map(stripNulls) as unknown as T;
  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, stripNulls(v)])
    ) as T;
  }
  return value;
};

export const jobToFormFields = (job: CustomizationJob): CustomizationFormFields => {
  if (isAutomodelJob(job)) {
    return {
      ...FORM_DEFAULTS,
      outputName: generateDefaultName(),
      description: job.description ?? '',
      backend: 'automodel',
      automodel: stripNulls(job.spec) as AutomodelJobInput,
    };
  }
  if (isRlJob(job)) {
    const spec = stripNulls(job.spec) as RlJobInput;
    // GRPO-only hyperparameters live in the `grpo` namespace, not on spec.training,
    // so replaying the spec alone would present a cloned GRPO job as a default DPO one.
    const grpo = spec.training.type === 'grpo' ? spec.training : undefined;
    const defaults = FORM_DEFAULTS.grpo;
    return {
      ...FORM_DEFAULTS,
      outputName: generateDefaultName(),
      description: job.description ?? '',
      backend: 'rl',
      rl: spec,
      grpo: {
        ...defaults,
        trainingType: grpo ? 'grpo' : 'dpo',
        environmentFileset: spec.environment ?? '',
        ...(grpo && {
          num_generations_per_prompt:
            grpo.num_generations_per_prompt ?? defaults.num_generations_per_prompt,
          num_prompts_per_step: grpo.num_prompts_per_step ?? defaults.num_prompts_per_step,
          overlong_filtering: grpo.overlong_filtering ?? defaults.overlong_filtering,
          max_rollout_turns: grpo.max_rollout_turns ?? defaults.max_rollout_turns,
          normalize_rewards: grpo.normalize_rewards ?? defaults.normalize_rewards,
          ratio_clip_min: grpo.ratio_clip_min ?? defaults.ratio_clip_min,
          ratio_clip_max: grpo.ratio_clip_max ?? defaults.ratio_clip_max,
          temperature: grpo.temperature ?? defaults.temperature,
          val_at_start: grpo.val_at_start ?? defaults.val_at_start,
          max_new_tokens: grpo.max_new_tokens ?? defaults.max_new_tokens,
          finetuning_type: grpo.finetuning_type ?? defaults.finetuning_type,
          lora: grpo.lora ? { ...defaults.lora, ...grpo.lora } : defaults.lora,
          use_dynamic_sampling: grpo.use_dynamic_sampling ?? defaults.use_dynamic_sampling,
          batch_multiplier: grpo.batch_multiplier ?? defaults.batch_multiplier,
          dynamic_sampling_max_gen_batches:
            grpo.dynamic_sampling_max_gen_batches ?? defaults.dynamic_sampling_max_gen_batches,
          use_leave_one_out_baseline:
            grpo.use_leave_one_out_baseline ?? defaults.use_leave_one_out_baseline,
          use_importance_sampling_correction:
            grpo.use_importance_sampling_correction ?? defaults.use_importance_sampling_correction,
          use_on_policy_kl_approximation:
            grpo.use_on_policy_kl_approximation ?? defaults.use_on_policy_kl_approximation,
          policy_backend: grpo.policy_backend ?? defaults.policy_backend,
          batching_strategy: grpo.batching_strategy ?? defaults.batching_strategy,
          sequence_length_round: grpo.sequence_length_round ?? defaults.sequence_length_round,
          vllm_gpu_memory_utilization:
            grpo.vllm_gpu_memory_utilization ?? defaults.vllm_gpu_memory_utilization,
          // No `??` fallback: these are unset by default, and absence is meaningful.
          ratio_clip_c: grpo.ratio_clip_c,
          advantage_clip_low: grpo.advantage_clip_low,
          advantage_clip_high: grpo.advantage_clip_high,
          truncated_importance_sampling_type: grpo.truncated_importance_sampling_type,
          truncated_importance_sampling_ratio: grpo.truncated_importance_sampling_ratio,
          truncated_importance_sampling_ratio_min: grpo.truncated_importance_sampling_ratio_min,
          top_k: grpo.top_k,
          train_mb_tokens: grpo.train_mb_tokens,
          router_aux_loss_coef: grpo.router_aux_loss_coef,
          vllm_tensor_parallel_size: grpo.vllm_tensor_parallel_size,
          reward_scaling: grpo.reward_scaling,
          reward_shaping: grpo.reward_shaping,
          hf_config_overrides: grpo.hf_config_overrides,
          automodel_kwargs: grpo.automodel_kwargs,
        }),
      },
    };
  }
  return {
    ...FORM_DEFAULTS,
    outputName: generateDefaultName(),
    description: job.description ?? '',
    backend: 'unsloth',
    unsloth: stripNulls(job.spec) as UnslothJobInput,
  };
};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Fill a partial set of form fields out to a complete one by merging over `FORM_DEFAULTS`.
 *
 * Used to seed the form from sources that are not guaranteed complete — notably saved job
 * templates, whose payload is opaque JSON in the entity-store and can therefore be missing
 * whole sections if it was written by an older Studio build. Merging keeps every input
 * controlled instead of flipping to uncontrolled mid-render.
 *
 * Deliberately *not* `customizationFormSchema` validation: that schema encodes
 * submit-readiness (a model and dataset must be chosen), which is a stricter bar than
 * "usable as a starting point". A half-configured template should still open the form
 * pre-filled and let the user finish it, rather than being silently discarded.
 *
 * Driven off `FORM_DEFAULTS`' own keys and the backend enum, so adding a backend section
 * needs no change here.
 */
/**
 * Merge a stored value over its default, recursing into nested objects so a partial
 * section keeps the defaults it omits. A one-level merge would replace whole sub-objects —
 * `rl.training` or `grpo.lora` arriving partial would drop every key it did not name.
 *
 * Arrays are replaced rather than merged: a stored list is the whole list.
 */
const mergeOverDefaults = (fallback: unknown, provided: unknown): unknown => {
  if (!isPlainObject(fallback) || !isPlainObject(provided)) return provided;
  const out: Record<string, unknown> = { ...fallback };
  for (const [key, value] of Object.entries(provided)) {
    if (value === undefined) continue;
    out[key] = mergeOverDefaults(fallback[key], value);
  }
  return out;
};

export const coerceToFormFields = (value: unknown): CustomizationFormFields | undefined => {
  if (!isPlainObject(value)) return undefined;
  if (!isCustomizationBackend(value.backend)) return undefined;

  const merged: Record<string, unknown> = { ...FORM_DEFAULTS };
  for (const [key, fallback] of Object.entries(FORM_DEFAULTS)) {
    const provided = value[key];
    if (provided === undefined) continue;
    merged[key] = mergeOverDefaults(fallback, provided);
  }

  const fields = merged as unknown as CustomizationFormFields;
  return fields.outputName ? fields : { ...fields, outputName: generateDefaultName() };
};

/**
 * Form values for a route entered with state: either a template supplying `initialValues`
 * outright, or a job to clone.
 */
export const getInitialFormValuesFromState = (
  state: unknown
): CustomizationFormFields | undefined => {
  if (typeof state !== 'object' || state === null) return undefined;
  const { initialValues, cloneFromJob } = state as {
    initialValues?: unknown;
    cloneFromJob?: unknown;
  };

  if (customizationFormSchema.safeParse(initialValues).success) {
    return initialValues as CustomizationFormFields;
  }

  if (typeof cloneFromJob === 'object' && cloneFromJob !== null) {
    const spec = (cloneFromJob as Record<string, unknown>).spec;
    if (isAutomodelSpec(spec) || isUnslothSpec(spec) || isRlSpec(spec)) {
      return jobToFormFields(cloneFromJob as CustomizationJob);
    }
  }

  return undefined;
};
