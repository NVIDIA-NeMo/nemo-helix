// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import type { DraftInputs } from '@studio/components/CreateCustomizationStart/aiDraft';
import {
  buildDraftMessages,
  buildRetryMessages,
  describeModel,
  SKILL_REFERENCES,
} from '@studio/components/CreateCustomizationStart/aiInstructions';
import {
  GYM_DATASET,
  HYBRID_MOE_MODEL,
  INPUTS,
  PREFERENCE_DATASET,
  SMALL_MODEL,
  SQL_ENVIRONMENT,
} from '@studio/components/CreateCustomizationStart/testFixtures';

describe('describeModel', () => {
  it('gives the facts the sizing guidance reasons about', () => {
    expect(describeModel(SMALL_MODEL)).toBe(
      'default/llama-8b — llama, 8.0B params, dense, chat/instruct, context 8192 tokens, min GPUs LoRA 1, min GPUs full-weight 4'
    );
  });

  it('calls out MoE and hybrid Mamba architectures', () => {
    const line = describeModel(HYBRID_MOE_MODEL);
    expect(line).toContain('MoE: 128 experts, 6 active per token');
    expect(line).toContain('hybrid Mamba-Transformer');
  });

  it('says the size is unknown when the model has no spec', () => {
    const bare = { id: 'x', name: 'mystery', workspace: 'default' } as ModelEntity;
    expect(describeModel(bare)).toMatch(/size unknown/);
  });
});

describe('messages', () => {
  it('keeps the instructions in the system turn and the picks with the goal in the user turn', () => {
    const [system, user] = buildDraftMessages('route tickets', INPUTS);
    expect(system.role).toBe('system');
    expect(system.content).toContain('# Fields Studio fills in');
    expect(system.content).not.toContain('# Inputs');
    expect(user.role).toBe('user');
    expect(user.content).toContain(`## Base model\n${describeModel(SMALL_MODEL)}`);
    expect(user.content).toContain('## Dataset\ndefault/tickets\n- Format: Chat Completion');
    expect(user.content).toContain('Training examples: 3000');
    expect(String(user.content).endsWith('# Goal\nroute tickets')).toBe(true);
  });

  it('shows the row shape without any of the data', () => {
    const [system, user] = buildDraftMessages('route tickets', INPUTS);
    expect(user.content).toContain('- Row shape (field names and types only):\n{\n  messages: [{');
    expect(system.content).toContain('never the data itself');
  });

  it('leaves out the shape when the dataset has no rows', () => {
    const [, user] = buildDraftMessages('route tickets', {
      ...INPUTS,
      dataset: { ...INPUTS.dataset, shape: '' },
    });
    expect(user.content).not.toContain('- Row shape');
  });

  it('says when the example count is an estimate', () => {
    const [, user] = buildDraftMessages('route tickets', {
      ...INPUTS,
      dataset: { ...INPUTS.dataset, trainingRowCount: 52000, rowCountIsEstimate: true },
    });
    expect(user.content).toContain('- Training examples: about 52000 (estimated from file sizes)');
  });

  it('says when no reward environment was picked', () => {
    const [, user] = buildDraftMessages('route tickets', INPUTS);
    expect(user.content).toContain('## Reward environment (grpo only)\nNot picked.');
  });

  it('describes a picked environment from its manifest', () => {
    const [, user] = buildDraftMessages('write sql', { ...INPUTS, environment: SQL_ENVIRONMENT });
    expect(user.content).toContain(
      '## Reward environment (grpo only)\ndefault/sql-env — text2sql: Executes generated SQL and rewards matching result sets.\n- Format: adapter-wheels-v1\n- Agent: verifiers_agent'
    );
  });

  it('embeds only the references for backends that can train the dataset, verbatim', () => {
    const referenced = (inputs: DraftInputs) => {
      const content = String(buildDraftMessages('goal', inputs)[0].content);
      return Object.keys(SKILL_REFERENCES).filter((file) =>
        content.includes(`# Reference: ${file}\n\n${SKILL_REFERENCES[file]}`)
      );
    };
    expect(referenced(INPUTS)).toEqual([
      'hyperparameters-automodel.md',
      'hyperparameters-unsloth.md',
      'batch-sizing.md',
    ]);
    expect(referenced({ ...INPUTS, dataset: GYM_DATASET })).toEqual([
      'hyperparameters-rl.md',
      'batch-sizing.md',
    ]);
    expect(referenced({ ...INPUTS, dataset: PREFERENCE_DATASET })).toEqual([
      'hyperparameters-rl.md',
      'batch-sizing.md',
    ]);
  });

  it('hands validation errors back after replaying the rejected job', () => {
    const first = buildDraftMessages('route tickets', INPUTS);
    const retry = buildRetryMessages(first, '{"backend":"automodel"}', [
      'training.lora_rank: not a field of the automodel job schema',
    ]);
    expect(retry.slice(0, first.length)).toEqual(first);
    expect(retry.at(-2)).toEqual({ role: 'assistant', content: '{"backend":"automodel"}' });
    expect(retry.at(-1)?.content).toContain(
      '- training.lora_rank: not a field of the automodel job schema'
    );
  });
});
