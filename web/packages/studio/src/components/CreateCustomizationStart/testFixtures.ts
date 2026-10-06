// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity, ModelSpec } from '@nemo/sdk/generated/platform/schema';
import type {
  CustomizationDraft,
  DraftDataset,
  DraftEnvironment,
  DraftInputs,
} from '@studio/components/CreateCustomizationStart/aiDraft';
import { CUSTOMIZER_SCHEMA_LABELS } from '@studio/util/customizerSchema';

const model = (name: string, spec: Partial<ModelSpec>): ModelEntity =>
  ({ id: name, name, workspace: 'default', spec }) as ModelEntity;

export const SMALL_MODEL = model('llama-8b', {
  family: 'llama',
  base_num_parameters: 8e9,
  is_chat: true,
  context_size: 8192,
  minimum_gpus_lora: 1,
  minimum_gpus_all_weights: 4,
});

export const HYBRID_MOE_MODEL = model('nemotron-moe', {
  family: 'nemotron_h',
  base_num_parameters: 30e9,
  is_chat: true,
  moe_config: { num_experts: 128, num_experts_per_tok: 6, num_expert_layers: 23 },
  mamba_config: { is_hybrid: true, num_mamba_layers: 23 },
});

const detection = (variant: NonNullable<DraftDataset['schema']>['variant']) => ({
  variant,
  label: CUSTOMIZER_SCHEMA_LABELS[variant],
});

const CHAT_DATASET: DraftDataset = {
  fileset: 'default/tickets',
  schema: detection('sft-chat'),
  trainingRowCount: 3000,
  rowCountIsEstimate: false,
  hasValidation: true,
  shape: '{\n  messages: [{\n    role: string,\n    content: string,\n  }],\n}',
};

export const PREFERENCE_DATASET: DraftDataset = {
  fileset: 'default/prefs',
  schema: detection('dpo-binary-preference'),
  trainingRowCount: 2000,
  rowCountIsEstimate: false,
  hasValidation: false,
  shape: '',
};

export const GYM_DATASET: DraftDataset = {
  ...PREFERENCE_DATASET,
  fileset: 'default/gym',
  schema: detection('grpo-gym'),
};

export const INPUTS: DraftInputs = {
  model: SMALL_MODEL,
  dataset: CHAT_DATASET,
  environment: null,
};

export const SQL_ENVIRONMENT: DraftEnvironment = {
  fileset: 'default/sql-env',
  manifest: {
    format: 'adapter-wheels-v1',
    agent: 'verifiers_agent',
    envName: 'text2sql',
    description: 'Executes generated SQL and rewards matching result sets.',
    wheelCount: 1,
  },
};

type Job = CustomizationDraft['job'];

/** An automodel SFT LoRA job, as the model would write it from the skill's template. */
export const automodelDraft = (
  job: Job = {},
  overrides: Partial<CustomizationDraft> = {}
): CustomizationDraft => ({
  backend: 'automodel',
  job: {
    model: 'placeholder',
    dataset: { training: 'placeholder' },
    training: {
      training_type: 'sft',
      finetuning_type: 'lora',
      lora: { rank: 16, alpha: 32 },
    },
    optimizer: { learning_rate: 1e-4 },
    output: { name: 'llama-8b-ticket-router' },
    ...job,
  },
  rationale: ['SFT, because your dataset holds chat transcripts.'],
  needs_from_user: [],
  ...overrides,
});
