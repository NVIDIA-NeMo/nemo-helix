// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import batchSizing from '@customizer-skill/references/batch-sizing.md?raw';
import automodel from '@customizer-skill/references/hyperparameters-automodel.md?raw';
import rl from '@customizer-skill/references/hyperparameters-rl.md?raw';
import unsloth from '@customizer-skill/references/hyperparameters-unsloth.md?raw';
import { getEntityReference } from '@nemo/common/src/namedEntity';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import {
  type DraftDataset,
  type DraftEnvironment,
  type DraftInputs,
  methodsForVariant,
} from '@studio/components/CreateCustomizationStart/aiDraft';
import { CUSTOMIZER_SCHEMA_LABELS } from '@studio/util/customizerSchema';
import type { ChatCompletionMessageParam, ChatCompletionTool } from 'openai/resources/index.mjs';

export const DRAFT_TOOL_NAME = 'draft_customization_job';

/**
 * The model writes the real job body for the backend it picks — the same JSON an agent
 * following the `nemo-customizer` skill submits — so the contract is the backend's job
 * schema, checked by the generated validators, rather than a summary Studio translates.
 */
export const draftCustomizationJobTool: ChatCompletionTool = {
  type: 'function',
  function: {
    name: DRAFT_TOOL_NAME,
    description:
      'Draft a fine-tuning job for the user to review: pick the backend, then write its complete job body.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['backend', 'job', 'rationale', 'needs_from_user'],
      properties: {
        backend: {
          type: 'string',
          enum: ['automodel', 'unsloth', 'rl'],
          description: 'Which backend plugin the job is for.',
        },
        job: {
          type: 'object',
          description:
            "The job body for that backend, shaped like the backend's full template in the references (AutomodelJobInput, UnslothJobInput, or RlJobInput). Only fields from that schema; unknown keys are rejected.",
        },
        rationale: {
          type: 'array',
          items: { type: 'string' },
          description: '2 to 5 plain-language sentences tying the key choices to the inputs.',
        },
        needs_from_user: {
          type: 'array',
          items: { type: 'string' },
          description: 'Steps the user must take before the job can run. Empty when complete.',
        },
      },
    },
  },
};

/**
 * The backend team's `nemo-customizer` skill references, verbatim. They are the knowledge an
 * agent drafting a job through the CLI works from, so the draft here works from the same
 * text — a change to the skill reaches Studio on the next build, with nothing to re-sync.
 */
export const SKILL_REFERENCES: Record<string, string> = {
  'hyperparameters-automodel.md': automodel,
  'hyperparameters-unsloth.md': unsloth,
  'hyperparameters-rl.md': rl,
  'batch-sizing.md': batchSizing,
};

/**
 * The knowledge is the skill's, verbatim (`SKILL_REFERENCES`). This brief covers only what
 * differs from the skill's CLI setting: what to ignore, what Studio fills in, how to answer.
 */
const SYSTEM_PROMPT = `You are a fine-tuning engineer drafting a NeMo Customizer job that the user will review in a form. Call the ${DRAFT_TOOL_NAME} tool exactly once with the backend you choose and its complete job body.

The user has picked the base model, the training dataset, and optionally a reward environment — all described under "Inputs" — and describes the goal of the fine-tune. Make the decisions an experienced engineer would: the method, the backend, the fine-tuning type, and every hyperparameter you have a reason to set. The backend team's references below are your source of truth for field names, valid values, full templates, and tuning guidance. The dataset's row shape lists its field names and value types — never the data itself; use it with the goal to understand the task.

# Using the references
- They are written for an agent driving the \`nemo\` CLI. Ignore everything about CLI commands, registering models or filesets, uploading data, submitting, polling, and files such as /tmp/job.json. You only write the job body.
- Shape \`job\` like the full template for the backend you pick. Every backend rejects keys its schema does not define.

# Fields Studio fills in
- The base model, dataset, and reward environment references are replaced with the user's picks; write any placeholder.
- Distillation's teacher_model is picked in the form: leave it out and add that step to needs_from_user. Do the same for a GRPO reward environment when none is listed under Inputs.

# Answer
- Name the output (output.name) after the task and the base: lowercase letters, digits, and hyphens, starting with a letter, at most 50 characters.
- Always call the tool. Never reply in plain text and never ask a question — make the most reasonable assumption and say so in rationale.`;

/** What the guidance sizes a job by: size, architecture, context, and GPU floor. */
export const describeModel = (model: ModelEntity): string => {
  const spec = model.spec;
  const facts = spec
    ? [
        spec.family,
        `${(spec.base_num_parameters / 1e9).toFixed(1)}B params`,
        spec.moe_config
          ? `MoE: ${spec.moe_config.num_experts} experts, ${spec.moe_config.num_experts_per_tok} active per token`
          : 'dense',
        spec.mamba_config?.is_hybrid ? 'hybrid Mamba-Transformer' : null,
        spec.is_chat ? 'chat/instruct' : 'base (not chat-tuned)',
        spec.context_size ? `context ${spec.context_size} tokens` : null,
        spec.minimum_gpus_lora ? `min GPUs LoRA ${spec.minimum_gpus_lora}` : null,
        spec.minimum_gpus_all_weights
          ? `min GPUs full-weight ${spec.minimum_gpus_all_weights}`
          : null,
      ].filter(Boolean)
    : ['no architecture metadata (size unknown)'];
  return `${getEntityReference(model)} — ${facts.join(', ')}`;
};

const describeDataset = ({
  fileset,
  schema,
  trainingRowCount,
  rowCountIsEstimate,
  hasValidation,
  shape,
}: DraftDataset) => {
  const format = schema
    ? `${CUSTOMIZER_SCHEMA_LABELS[schema.variant]} (${schema.variant}) — valid for method ${methodsForVariant(
        schema.variant
      )
        .map((m) => `"${m}"`)
        .join(' or ')}`
    : 'unknown';
  return [
    fileset,
    `- Format: ${format}`,
    `- Training examples: ${rowCountIsEstimate ? `about ${trainingRowCount} (estimated from file sizes)` : trainingRowCount}`,
    `- Validation split: ${hasValidation ? 'provided' : 'none (10% of training is held out automatically)'}`,
    ...(shape ? [`- Row shape (field names and types only):\n${shape}`] : []),
  ].join('\n');
};

const describeEnvironment = (environment: DraftEnvironment | null): string => {
  if (!environment) return 'Not picked.';
  const { fileset, manifest } = environment;
  if (!manifest) return fileset;
  const about = [manifest.envName, manifest.description].filter(Boolean).join(': ');
  return [
    `${fileset} — ${about}`,
    `- Format: ${manifest.format}`,
    ...(manifest.agent ? [`- Agent: ${manifest.agent}`] : []),
  ].join('\n');
};

const buildSystemPrompt = ({ model, dataset, environment }: DraftInputs): string =>
  [
    SYSTEM_PROMPT,
    '# Inputs',
    `## Base model\n${describeModel(model)}`,
    `## Dataset\n${describeDataset(dataset)}`,
    `## Reward environment (grpo only)\n${describeEnvironment(environment)}`,
    ...Object.entries(SKILL_REFERENCES).map(
      ([file, content]) => `# Reference: ${file}\n\n${content}`
    ),
  ].join('\n\n');

export const buildDraftMessages = (
  prompt: string,
  inputs: DraftInputs
): ChatCompletionMessageParam[] => [
  { role: 'system', content: buildSystemPrompt(inputs) },
  { role: 'user', content: prompt },
];

/**
 * Hands the validation errors back, the way the skill's agent reads a rejected submit and
 * fixes the JSON. The previous draft is replayed as a plain assistant turn: a `tool_calls`
 * turn needs a matching `tool` reply, which providers enforce inconsistently.
 */
export const buildRetryMessages = (
  messages: ChatCompletionMessageParam[],
  draft: string,
  errors: string[]
): ChatCompletionMessageParam[] => [
  ...messages,
  { role: 'assistant', content: draft },
  {
    role: 'user',
    content: [
      'That job failed validation:',
      ...errors.map((error) => `- ${error}`),
      '',
      `Call ${DRAFT_TOOL_NAME} again with the corrected job. Change only what fixes these errors.`,
    ].join('\n'),
  },
];
