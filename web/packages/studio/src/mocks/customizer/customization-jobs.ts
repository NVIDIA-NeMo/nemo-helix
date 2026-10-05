// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getURNFromNamedEntityRef } from '@nemo/common/src/namedEntity';
import type { RlGRPOTraining } from '@nemo/sdk/generated/customizer/schema';
import { dataset } from '@studio/mocks/datasets';
import type { AutomodelJob, RlJob, UnslothJob } from '@studio/util/customizationBackend';

const datasetUri = getURNFromNamedEntityRef(dataset)!;

const completedStatusDetails = {
  phase: 'completed',
  step: 10,
  max_steps: 10,
  num_epochs: 1,
  epoch: 1,
  percentage_done: 100,
  train_loss: 0.9,
  val_loss: 0.9,
  train_lr: 0.000005,
  train_grad_norm: 1.2345,
  checkpoint_path: 'default/output-fileset/checkpoints/step-10',
  metrics: {
    train_loss: [
      { value: 0.15, step: 2, epoch: 1 },
      { value: 0.35, step: 4, epoch: 1 },
      { value: 0.55, step: 6, epoch: 1 },
      { value: 0.85, step: 8, epoch: 1 },
      { value: 0.9, step: 10, epoch: 1 },
    ],
    val_loss: [
      { value: 0.5, step: 2, epoch: 1 },
      { value: 0.6, step: 4, epoch: 1 },
      { value: 0.7, step: 6, epoch: 1 },
      { value: 0.8, step: 8, epoch: 1 },
      { value: 0.9, step: 10, epoch: 1 },
    ],
  },
  status_logs: [
    { updated_at: '2025-10-24T15:13:17', message: 'created' },
    {
      updated_at: '2025-10-24T15:13:17.175399',
      message: 'TrainingJobPending',
      detail: 'The training job is pending',
    },
    { updated_at: '2025-10-24T15:13:33', message: 'TrainingJobRunning' },
    { updated_at: '2025-10-24T15:16:18', message: 'TrainingJobCompleted' },
  ],
};

/** Automodel distillation job. */
export const customizationJob1: AutomodelJob = {
  id: 'cust-4k8XJ8fRYtQT8NTBbjxAqk',
  name: 'meta-llama-3.2-1b-distillation-job',
  created_at: '2025-06-25T21:41:02.067430',
  updated_at: '2025-06-25T21:42:14.242833',
  workspace: 'default',
  project: 'default/project-QRpQtqLB4CJ2fUxKSCWsFX',
  ownership: { created_by: '', access_policies: {} },
  description: 'This is a test customization job',
  spec: {
    model: 'meta/llama-3.2-1b-distillation@v1.0.0+A100',
    dataset: { training: datasetUri },
    training: {
      training_type: 'distillation',
      finetuning_type: 'lora',
      lora: { rank: 16, alpha: 32, dropout: 0, merge: false, target_modules: ['q_proj', 'v_proj'] },
      max_seq_length: 2048,
      precision: 'bf16',
      teacher_model: 'qwen/qwen-2_5-72b-instruct',
      teacher_precision: 'bf16',
      distillation_ratio: 0.5,
      distillation_temperature: 2,
      offload_teacher: false,
    },
    schedule: { epochs: 1, max_steps: 1000, seed: 42 },
    batch: { global_batch_size: 8, micro_batch_size: 1, sequence_packing: false },
    optimizer: {
      learning_rate: 0.0001,
      weight_decay: 0.01,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      warmup_steps: 100,
    },
    parallelism: {
      num_nodes: 1,
      num_gpus_per_node: 1,
      tensor_parallel_size: 1,
      pipeline_parallel_size: 1,
      context_parallel_size: 1,
      sequence_parallel: false,
    },
    output: {
      name: 'default/meta-llama-3.2-1b-instruct-distillation@cust-4k8XJ8fRYtQT8NTBbjxAqk',
      type: 'model',
      fileset: 'default/output-fileset',
    },
  },
  status: 'completed',
  status_details: completedStatusDetails,
};

/** Automodel SFT + LoRA job. */
export const customizationJob2: AutomodelJob = {
  id: 'cust-DTDYY777TapJkJwkq6jMDD',
  name: 'meta-llama-3.1-8b-sft-lora-job',
  created_at: '2025-06-04T19:10:17.026494',
  updated_at: '2025-06-04T19:15:26.480239',
  workspace: 'default',
  project: 'default/project-QRpQtqLB4CJ2fUxKSCWsFX',
  ownership: { created_by: '', access_policies: {} },
  spec: {
    model: 'meta/llama-3.1-8b-instruct@v1.0.0+A100',
    dataset: { training: datasetUri },
    training: {
      training_type: 'sft',
      finetuning_type: 'lora',
      lora: {
        rank: 32,
        alpha: 16,
        dropout: 0.1,
        merge: false,
        target_modules: ['q_proj', 'v_proj'],
      },
      max_seq_length: 2048,
    },
    schedule: { epochs: 1, max_steps: 1000, seed: 42 },
    batch: { global_batch_size: 8, micro_batch_size: 1, sequence_packing: false },
    optimizer: {
      learning_rate: 0.0001,
      weight_decay: 0.01,
      adam_beta1: 0.9,
      adam_beta2: 0.999,
      warmup_steps: 0,
    },
    parallelism: {
      num_nodes: 1,
      num_gpus_per_node: 1,
      tensor_parallel_size: 1,
      pipeline_parallel_size: 1,
      context_parallel_size: 1,
      sequence_parallel: false,
    },
    output: {
      name: 'default/meta-llama-3.1-8b-instruct-academic-spoonbill-lora@cust-DTDYY777TapJkJwkq6jMDD',
      type: 'adapter',
      fileset: 'default/output-fileset',
    },
  },
  status: 'completed',
  status_details: { phase: 'completed', step: 44, max_steps: 44, epoch: 1, percentage_done: 100 },
};

/** Unsloth SFT + LoRA job. */
export const customizationJob3: UnslothJob = {
  id: 'cust-7hyykExVYdj9j8wMg6UKe2',
  name: 'meta-llama-3.1-8b-unsloth-sft-lora-job',
  created_at: '2025-06-04T19:10:16.633103',
  updated_at: '2025-06-04T19:34:26.406896',
  workspace: 'default',
  project: 'default/project-QRpQtqLB4CJ2fUxKSCWsFX',
  ownership: { created_by: '', access_policies: {} },
  spec: {
    model: {
      name: 'meta/llama-3.1-8b-instruct@v1.0.0+A100',
      max_seq_length: 2048,
      load_in_4bit: true,
      load_in_8bit: false,
      dtype: 'auto',
      trust_remote_code: false,
    },
    dataset: { path: datasetUri, text_field: 'text', apply_chat_template: false, packing: false },
    training: {
      training_type: 'sft',
      finetuning_type: 'lora',
      lora: {
        rank: 16,
        alpha: 16,
        dropout: 0,
        target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias: 'none',
        use_rslora: false,
        random_state: 3407,
      },
      use_gradient_checkpointing: 'unsloth',
    },
    schedule: {
      epochs: 3,
      warmup_steps: 5,
      lr_scheduler_type: 'linear',
      logging_steps: 1,
      seed: 3407,
    },
    batch: { per_device_train_batch_size: 2, gradient_accumulation_steps: 4 },
    optimizer: { learning_rate: 0.0002, weight_decay: 0, optim: 'adamw_8bit' },
    hardware: { gpus: '0', precision: 'bf16' },
    output: {
      name: 'default/meta-llama-3.1-8b-instruct-unsloth-lora@cust-7hyykExVYdj9j8wMg6UKe2',
      type: 'adapter',
      save_method: 'lora',
      fileset: 'default/output-fileset',
    },
  },
  status: 'completed',
  status_details: { phase: 'completed', step: 44, max_steps: 44, epoch: 3, percentage_done: 100 },
};

const TRAINING_STEPS = [1, 100, 200, 300, 400, 500];

const series = (values: number[]) =>
  values.map((value, index) => ({ value, step: TRAINING_STEPS[index], epoch: 1 }));

/**
 * Names taken from the golden test pinning NeMo-RL's logged dict, since the whole GRPO overview
 * reads through them.
 */
export const grpoStatusDetails = {
  phase: 'completed',
  step: 500,
  max_steps: 500,
  num_epochs: 1,
  epoch: 1,
  percentage_done: 100,
  checkpoint_path: 'default/grpo-output-fileset/checkpoints/step-500',
  train_reward: 0.617,
  val_accuracy: 0.58,
  train_truncation_rate: 0.041,
  train_sampling_importance_ratio: 1.002,
  'train_timing/total_step_time': 35.4,
  'train_total_reward/mean': 0.617,
  // Pinned to what the backend's golden dict reports, degenerate values and all: on a 0/1 reward
  // the quartiles land on 0 and 1 and the stddev is √(0.617·0.383).
  'train_total_reward/stddev': 0.486,
  'train_total_reward/p25': 0,
  'train_total_reward/p75': 1,
  metrics: {
    train_loss: [],
    val_loss: [],
    train_reward: series([0.18, 0.34, 0.45, 0.52, 0.58, 0.617]),
    val_accuracy: [
      { value: 0.31, step: 100, epoch: 1 },
      { value: 0.49, step: 300, epoch: 1 },
      { value: 0.58, step: 500, epoch: 1 },
    ],
    train_truncation_rate: series([0.09, 0.072, 0.061, 0.05, 0.044, 0.041]),
    train_gen_kl_error: series([0.00012, 0.00028, 0.00039, 0.00047, 0.00051, 0.00054]),
    train_token_mult_prob_error: series([1.001, 1.002, 1.003, 1.004, 1.004, 1.004]),
    train_sampling_importance_ratio: series([1.0004, 1.0009, 1.0013, 1.0017, 1.0019, 1.002]),
    train_approx_entropy: series([0.98, 0.74, 0.55, 0.43, 0.36, 0.31]),
    // Rises with the responses: longer generations cost more wall clock per step.
    'train_timing/total_step_time': series([29.8, 31.6, 33.6, 35, 34.8, 35.4]),
    'train_advantages/mean': series([0.001, 0.002, 0.003, 0.002, 0.004, 0.003]),
    train_kl_penalty: series([0, 0, 0, 0, 0, 0]),
    'train_gen_tokens_per_sample/mean': series([412.5, 498.1, 561.4, 612.9, 664.2, 689.3]),
  },
};

const grpoTraining: RlGRPOTraining = {
  type: 'grpo',
  finetuning_type: 'all_weights',
  epochs: 1,
  max_steps: 500,
  batch_size: 64,
  micro_batch_size: 1,
  max_seq_length: 4096,
  learning_rate: 0.000001,
  num_generations_per_prompt: 8,
  num_prompts_per_step: 8,
  temperature: 1,
  normalize_rewards: true,
  overlong_filtering: true,
  max_rollout_turns: 1,
  ref_policy_kl_penalty: 0,
  parallelism: { num_nodes: 1, num_gpus_per_node: 8, tensor_parallel_size: 4 },
};

/** NeMo-RL GRPO job against a NeMo Gym environment. */
export const grpoCustomizationJob: RlJob = {
  id: 'cust-9pQm2VxTgHs4RbNkLd7Wce',
  name: 'grpo-qwen-math-0812',
  created_at: '2026-08-12T14:02:11.412000',
  updated_at: '2026-08-12T19:48:52.901000',
  workspace: 'default',
  project: 'default/project-QRpQtqLB4CJ2fUxKSCWsFX',
  ownership: { created_by: '', access_policies: {} },
  description: 'GRPO against the math verifier environment',
  spec: {
    model: 'qwen/qwen2.5-7b-instruct',
    dataset: datasetUri,
    environment: 'default/math-verifier-env',
    training: grpoTraining,
    output: {
      name: 'default/qwen2.5-7b-instruct-grpo-math@cust-9pQm2VxTgHs4RbNkLd7Wce',
      type: 'model',
      fileset: 'default/grpo-output-fileset',
    },
  },
  status: 'completed',
  status_details: grpoStatusDetails,
};

/**
 * The mapped message the training runner reports on an OOM, taken from the `CudaError` rule in
 * `services/rl/src/nhx/rl/tasks/training/errors/error_rules.yaml`.
 */
export const grpoCudaErrorMessage =
  'Your job ran out of GPU memory during training. To reduce memory usage: 1) Reduce ' +
  'micro_batch_size, 2) Reduce max_seq_length, 3) Use LoRA/PEFT instead of all_weights ' +
  'fine-tuning, 4) Increase tensor_parallel_size to shard the model across more GPUs, or ' +
  '5) Request GPUs with more memory.';

/** GRPO run that died partway through training: reward series stops well short of max_steps. */
const failedGrpoStatusDetails = {
  phase: 'training',
  step: 300,
  max_steps: 500,
  num_epochs: 1,
  epoch: 0,
  percentage_done: 60,
  train_reward: 0.45,
  train_truncation_rate: 0.061,
  'train_timing/total_step_time': 33.6,
  metrics: {
    train_loss: [],
    val_loss: [],
    train_reward: series([0.18, 0.34, 0.45]),
    val_accuracy: [
      { value: 0.31, step: 100, epoch: 1 },
      { value: 0.49, step: 300, epoch: 1 },
    ],
    train_truncation_rate: series([0.09, 0.072, 0.061]),
    'train_timing/total_step_time': series([29.8, 31.6, 33.6]),
  },
  // What the Kubernetes backend leaves on the job. Deliberately useless: the resolver has to
  // reach into the step/task tree to find the mapped cause.
  message: 'One or more tasks are in error state',
};

/** GRPO job that failed during training with a mapped `CudaError`. */
export const failedGrpoCustomizationJob: RlJob = {
  ...grpoCustomizationJob,
  id: 'cust-2Hs7VnKp4RtQmXd9Lb3Wfe',
  name: 'grpo-qwen-math-0819-oom',
  created_at: '2026-08-19T09:14:03.220000',
  updated_at: '2026-08-19T10:02:41.775000',
  description: 'GRPO against the math verifier environment',
  status: 'error',
  status_details: failedGrpoStatusDetails,
  error_details: { message: 'One or more tasks are in error state' },
};

export const customizationJobs = [
  customizationJob1,
  customizationJob2,
  customizationJob3,
  grpoCustomizationJob,
  failedGrpoCustomizationJob,
];

export const customizationJobSteps = [
  {
    id: 'step-download',
    name: 'model-and-dataset-download',
    status: 'completed',
    status_details: { message: 'completed' },
    error_details: {},
    tasks: [],
    created_at: '2025-06-25T21:41:02.100000',
    updated_at: '2025-06-25T21:41:12.100000',
  },
  {
    id: 'step-training',
    name: 'training',
    status: 'completed',
    status_details: { message: 'completed' },
    error_details: {},
    tasks: [],
    created_at: '2025-06-25T21:41:12.100000',
    updated_at: '2025-06-25T21:42:02.100000',
  },
  {
    id: 'step-upload',
    name: 'model-upload',
    status: 'completed',
    status_details: { message: 'completed' },
    error_details: {},
    tasks: [],
    created_at: '2025-06-25T21:42:02.100000',
    updated_at: '2025-06-25T21:42:10.100000',
  },
  {
    id: 'step-entity',
    name: 'model-entity-creation',
    status: 'completed',
    status_details: { message: 'completed' },
    error_details: {},
    tasks: [],
    created_at: '2025-06-25T21:42:10.100000',
    updated_at: '2025-06-25T21:42:14.242833',
  },
];

/**
 * Single-node GRPO failure as the Kubernetes backend leaves it: the mapped `CudaError` sits on
 * the failing task, while the step and job above it carry only generic infrastructure text.
 */
export const failedGrpoJobSteps = [
  {
    id: 'step-download',
    name: 'model-dataset-environment-download',
    status: 'completed',
    status_details: { message: 'completed' },
    error_details: {},
    tasks: [],
    created_at: '2026-08-19T09:14:10.000000',
    updated_at: '2026-08-19T09:21:44.000000',
  },
  {
    id: 'step-training',
    name: 'grpo-training',
    status: 'error',
    status_details: { message: 'Job has errored pods, check tasks for error details' },
    error_details: { message: 'One or more tasks are in error state' },
    tasks: [
      {
        id: 'task-7c1f9d2a-4b8e-4f31-9a2c-6d5e8b3a1c07',
        name: 'task-7c1f9d2a-4b8e-4f31-9a2c-6d5e8b3a1c07',
        status: 'error',
        status_details: { phase: 'training', step: 300 },
        error_details: {
          message: grpoCudaErrorMessage,
          type: 'CudaError',
          detail: 'OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB',
        },
        error_stack:
          'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 0 has a total capacity of 79.15 GiB of which 412.00 MiB is free.',
        created_at: '2026-08-19T09:21:50.000000',
        updated_at: '2026-08-19T10:02:38.000000',
      },
    ],
    created_at: '2026-08-19T09:21:44.000000',
    updated_at: '2026-08-19T10:02:41.775000',
  },
];

/**
 * Multi-node GRPO failure. The Volcano backend never passes `error_details` when building a step
 * update, so the only text anywhere in the tree is the step's `status_details.message`.
 */
export const failedVolcanoGrpoJobSteps = [
  {
    id: 'step-training',
    name: 'grpo-training',
    status: 'error',
    status_details: { message: 'Job failed' },
    error_details: null,
    tasks: [],
    created_at: '2026-08-19T09:21:44.000000',
    updated_at: '2026-08-19T10:02:41.775000',
  },
];
