// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  customizationJobToConfigJson,
  parseCustomizationJson,
} from '@studio/components/CreateCustomizationStart/jsonConfig';
import { CustomizationBackend } from '@studio/util/customizationBackend';
import { AUTOMODEL_DEFAULT_SPEC } from '@studio/util/forms/specDefaults';

const automodelSpec = {
  model: 'ws/llama-3.2-1b',
  dataset: { training: 'ws/my-dataset' },
  training: { finetuning_type: 'lora' },
};

const unslothSpec = {
  model: { name: 'ws/llama-3.2-1b' },
  dataset: { path: 'ws/my-dataset' },
  hardware: {},
};

const dpoSpec = {
  model: 'ws/llama-3.2-1b',
  dataset: 'ws/my-prefs',
  training: { type: 'dpo' },
};

/** Narrows to the success arm so tests can read `fields` without repeating the guard. */
const expectOk = (result: ReturnType<typeof parseCustomizationJson>) => {
  if (!result.ok) throw new Error(`expected a parse, got: ${result.error}`);
  return result;
};

describe('parseCustomizationJson', () => {
  it.each([
    ['a bare spec', automodelSpec],
    ['a full job request', { spec: automodelSpec }],
  ])('accepts %s', (_label, payload) => {
    const result = expectOk(parseCustomizationJson(JSON.stringify(payload)));
    expect(result.backend).toBe(CustomizationBackend.automodel);
    expect(result.fields.automodel.model).toBe('ws/llama-3.2-1b');
  });

  it.each([
    ['automodel', automodelSpec, CustomizationBackend.automodel],
    ['unsloth', unslothSpec, CustomizationBackend.unsloth],
    ['rl', dpoSpec, CustomizationBackend.rl],
  ])('detects a %s config and sets the form backend to match', (_label, spec, backend) => {
    const result = expectOk(parseCustomizationJson(JSON.stringify(spec)));
    expect(result.backend).toBe(backend);
    expect(result.fields.backend).toBe(backend);
  });

  it('reads GRPO off the training discriminator, not just the RL shape', () => {
    const result = expectOk(
      parseCustomizationJson(
        // GRPO is the one arm the schema requires a reward environment for.
        JSON.stringify({
          model: 'ws/m',
          dataset: 'ws/d',
          environment: 'ws/my-env',
          training: { type: 'grpo' },
        })
      )
    );
    expect(result.backend).toBe(CustomizationBackend.rl);
    expect(result.fields.grpo.trainingType).toBe('grpo');
    expect(result.fields.grpo.environmentFileset).toBe('ws/my-env');
  });

  it('fills unmentioned fields from the backend defaults', () => {
    const result = expectOk(parseCustomizationJson(JSON.stringify(automodelSpec)));
    // The config says nothing about the optimizer, so it must arrive fully defaulted.
    expect(result.fields.automodel.optimizer).toEqual(AUTOMODEL_DEFAULT_SPEC.optimizer);
  });

  it('keeps values the config does set, at every nesting depth', () => {
    const result = expectOk(
      parseCustomizationJson(
        JSON.stringify({
          ...automodelSpec,
          training: { finetuning_type: 'lora', lora: { rank: 64 } },
          optimizer: { learning_rate: 0.0002 },
        })
      )
    );
    expect(result.fields.automodel.training.lora?.rank).toBe(64);
    expect(result.fields.automodel.optimizer?.learning_rate).toBe(0.0002);
    // Merging `lora.rank` must not blow away its sibling defaults.
    expect(result.fields.automodel.training.lora?.alpha).toBe(
      AUTOMODEL_DEFAULT_SPEC.training.lora?.alpha
    );
  });

  it('replaces arrays wholesale rather than merging element-wise', () => {
    const result = expectOk(
      parseCustomizationJson(
        JSON.stringify({
          ...automodelSpec,
          training: { finetuning_type: 'lora', lora: { exclude_modules: ['*.out_proj'] } },
        })
      )
    );
    expect(result.fields.automodel.training.lora?.exclude_modules).toEqual(['*.out_proj']);
  });

  it('takes the output name from the request when it names one', () => {
    const result = expectOk(
      parseCustomizationJson(JSON.stringify({ name: 'my-model', spec: automodelSpec }))
    );
    expect(result.fields.outputName).toBe('my-model');
  });

  it('generates an output name when the config does not carry one', () => {
    const result = expectOk(parseCustomizationJson(JSON.stringify(automodelSpec)));
    expect(result.fields.outputName).not.toBe('');
  });

  it('carries the description through', () => {
    const result = expectOk(
      parseCustomizationJson(JSON.stringify({ description: 'nightly run', spec: automodelSpec }))
    );
    expect(result.fields.description).toBe('nightly run');
  });

  it.each([
    ['empty input', '   '],
    ['malformed JSON', '{ "model": '],
    ['a JSON array', '[]'],
    ['a bare value', '42'],
    ['an unrecognisable object', '{ "foo": "bar" }'],
  ])('rejects %s with a message instead of throwing', (_label, input) => {
    const result = parseCustomizationJson(input);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).not.toBe('');
  });

  it('rejects a GRPO config with no reward environment, naming the field', () => {
    const result = parseCustomizationJson(
      JSON.stringify({ model: 'ws/m', dataset: 'ws/d', training: { type: 'grpo' } })
    );
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain('environmentFileset');
  });

  it('rejects a config whose values violate the schema', () => {
    const result = parseCustomizationJson(
      JSON.stringify({ ...automodelSpec, optimizer: { learning_rate: -1 } })
    );
    expect(result.ok).toBe(false);
  });
});

describe('customizationJobToConfigJson', () => {
  const job = {
    id: 'cust-123',
    name: 'my-job',
    description: 'a run',
    status: 'completed',
    created_at: '2026-01-01T00:00:00Z',
    spec: {
      model: 'default/qwen3-0-6b',
      dataset: { training: 'default/hellaswag-demo' },
      training: { finetuning_type: 'lora', lora: { rank: 64 } },
    },
  } as unknown as Parameters<typeof customizationJobToConfigJson>[0];

  it('keeps only what a new job can be started from', () => {
    const written = JSON.parse(customizationJobToConfigJson(job)) as Record<string, unknown>;

    expect(Object.keys(written).sort()).toEqual(['description', 'name', 'spec']);
    expect(written.id).toBeUndefined();
    expect(written.status).toBeUndefined();
    expect(written.created_at).toBeUndefined();
  });

  /** The two ends of this feature: whatever a job writes out has to load straight back in. */
  it('round-trips through the parser it feeds', () => {
    const result = parseCustomizationJson(customizationJobToConfigJson(job));

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.outputName).toBe('my-job');
    }
  });
});
