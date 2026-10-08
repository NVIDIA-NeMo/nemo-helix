// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type CustomizationDraft,
  type DraftInputs,
  type DraftValidation,
  datasetProblem,
  ERROR_NOT_JSON,
  estimateTrainingRows,
  NO_TRAINING_FILES,
  UNRECOGNIZED_FORMAT,
  validateDraft,
} from '@studio/components/CreateCustomizationStart/aiDraft';
import {
  automodelDraft,
  GYM_DATASET,
  INPUTS,
  PREFERENCE_DATASET,
  SQL_ENVIRONMENT,
} from '@studio/components/CreateCustomizationStart/testFixtures';

const expectValid = (result: DraftValidation) => {
  if (result.status !== 'valid') throw new Error(`expected valid, got ${result.errors}`);
  return result;
};

const expectErrors = (result: DraftValidation) => {
  if (result.status !== 'invalid') throw new Error('expected invalid');
  return result.errors;
};

const validate = (draft: CustomizationDraft, inputs: DraftInputs = INPUTS) =>
  validateDraft(JSON.stringify(draft), inputs);

describe('reading the tool call', () => {
  it('rejects a reply that is not JSON', () => {
    expect(validateDraft('not json', INPUTS)).toEqual({
      status: 'invalid',
      errors: [ERROR_NOT_JSON],
    });
  });

  it('names what is missing, so a retry can target it', () => {
    expect(expectErrors(validateDraft(JSON.stringify({ job: {} }), INPUTS))[0]).toMatch(
      /^backend:/
    );
  });
});

describe('validateDraft', () => {
  describe("the user's picks", () => {
    it('replaces whatever references the model wrote', () => {
      const { values } = expectValid(validate(automodelDraft()));
      expect(values.automodel.model).toBe('default/llama-8b');
      expect(values.automodel.dataset).toMatchObject({
        training: 'default/tickets',
        validation: 'default/tickets',
      });
    });

    it('leaves the teacher for the user to pick in the form', () => {
      const draft = automodelDraft({
        training: { training_type: 'distillation', teacher_model: 'default/made-up' },
      });
      const { values, summary } = expectValid(validate(draft));
      expect(values.automodel.training.teacher_model).toBeUndefined();
      expect(summary.method).toBe('distillation');
    });

    it('fills in the picked reward environment for GRPO', () => {
      const draft: CustomizationDraft = {
        backend: 'rl',
        job: {
          model: 'x',
          dataset: 'x',
          training: { type: 'grpo', learning_rate: 1e-6, num_generations_per_prompt: 8 },
          output: { name: 'math-grpo' },
        },
        rationale: [],
        needs_from_user: [],
      };
      const { values, summary } = expectValid(
        validate(draft, { ...INPUTS, dataset: GYM_DATASET, environment: SQL_ENVIRONMENT })
      );
      expect(values.backend).toBe('rl');
      expect(values.grpo).toMatchObject({
        trainingType: 'grpo',
        environmentFileset: 'default/sql-env',
        num_generations_per_prompt: 8,
      });
      expect(summary.environment).toBe('default/sql-env');
    });
  });

  describe('the job schema', () => {
    it('reports a key the backend would reject, instead of dropping it', () => {
      const draft = automodelDraft({ training: { training_type: 'sft', lora_rank: 8 } });
      expect(expectErrors(validate(draft))).toEqual([
        'training.lora_rank: not a field of the automodel job schema',
      ]);
    });

    it("accepts the skill template's nulls as unset fields", () => {
      const draft = automodelDraft({
        schedule: { epochs: 1, max_steps: null, val_check_interval: null, seed: null },
        training: { training_type: 'sft', finetuning_type: 'lora', precision: null },
      });
      const { values, summary } = expectValid(validate(draft));
      expect(values.automodel.schedule?.max_steps).toBeUndefined();
      expect(summary.settings.map((s) => s.label)).not.toContain('schedule.max_steps');
    });

    it('reports an invalid value by its path', () => {
      const draft = automodelDraft({ optimizer: { learning_rate: -1 } });
      expect(expectErrors(validate(draft))[0]).toMatch(/^optimizer\.learning_rate:/);
    });

    it('requires the RL training type, which picks the DPO or GRPO arm', () => {
      const draft: CustomizationDraft = {
        backend: 'rl',
        job: { training: { learning_rate: 5e-6 } },
        rationale: [],
        needs_from_user: [],
      };
      expect(expectErrors(validate(draft, { ...INPUTS, dataset: PREFERENCE_DATASET }))[0]).toMatch(
        /^training/
      );
    });

    it("rejects a method the dataset's format cannot train", () => {
      expect(
        expectErrors(validate(automodelDraft(), { ...INPUTS, dataset: PREFERENCE_DATASET }))
      ).toEqual([
        'The dataset is in Binary Preference format, which only works with dpo, but the job trains sft.',
      ]);
    });
  });

  describe('into the form', () => {
    it('keeps every value the model chose, hardware included', () => {
      const draft = automodelDraft({
        schedule: { epochs: 3 },
        batch: { global_batch_size: 64, micro_batch_size: 16 },
        parallelism: { num_gpus_per_node: 2 },
      });
      const { automodel } = expectValid(validate(draft)).values;
      expect(automodel.training.lora).toMatchObject({ rank: 16, alpha: 32 });
      expect(automodel.optimizer?.learning_rate).toBe(1e-4);
      expect(automodel.schedule?.epochs).toBe(3);
      expect(automodel.batch).toMatchObject({ global_batch_size: 64, micro_batch_size: 16 });
      expect(automodel.parallelism?.num_gpus_per_node).toBe(2);
    });

    it("builds the full config with the form's own request builder", () => {
      const { config } = expectValid(validate(automodelDraft()));
      const request = JSON.parse(config);
      expect(request.name).toBe('llama-8b-ticket-router');
      expect(request.spec).toMatchObject({
        model: 'default/llama-8b',
        dataset: { training: 'default/tickets' },
        optimizer: { learning_rate: 1e-4 },
      });
      // Defaults the model never wrote are filled in, so the config is the whole job.
      expect(request.spec.optimizer.weight_decay).toBeDefined();
    });

    it('takes the output name from the job', () => {
      expect(expectValid(validate(automodelDraft())).values.outputName).toBe(
        'llama-8b-ticket-router'
      );
    });

    it('keeps an unsloth merged save', () => {
      const draft: CustomizationDraft = {
        backend: 'unsloth',
        job: {
          model: { name: 'x' },
          dataset: { path: 'x' },
          training: { finetuning_type: 'lora' },
          optimizer: { learning_rate: 2e-4 },
          output: { name: 'merged-run', save_method: 'merged_16bit' },
        },
        rationale: [],
        needs_from_user: [],
      };
      const { values } = expectValid(validate(draft));
      expect(values.unsloth.model.name).toBe('default/llama-8b');
      expect(values.unsloth.output?.save_method).toBe('merged_16bit');
    });
  });

  describe('summary', () => {
    it('lists the values the model wrote, by path, and none of the references', () => {
      const { summary } = expectValid(validate(automodelDraft()));
      expect(summary.settings).toEqual([
        { label: 'training.training_type', value: 'sft' },
        { label: 'training.finetuning_type', value: 'lora' },
        { label: 'training.lora.rank', value: '16' },
        { label: 'training.lora.alpha', value: '32' },
        { label: 'optimizer.learning_rate', value: '1e-4' },
      ]);
    });
  });
});

describe('estimateTrainingRows', () => {
  const file = (rowCount: number, size: number, bytesRead: number) =>
    ({ path: 'training.jsonl', file_ref: '', file_url: '', size, rowCount, bytesRead }) as const;

  it('counts exactly when every file was read whole', () => {
    expect(estimateTrainingRows([file(100, 1_000, 1_000), file(50, 500, 500)])).toEqual({
      trainingRowCount: 150,
      rowCountIsEstimate: false,
    });
  });

  it('keeps a whole-file count as is, however large the file', () => {
    expect(estimateTrainingRows([file(80_000, 10_000_000, 10_000_000)])).toEqual({
      trainingRowCount: 80_000,
      rowCountIsEstimate: false,
    });
  });

  it('never presents the count as exact when the files are not valid UTF-8', () => {
    expect(estimateTrainingRows([file(100, 1_000, 1_000)], false)).toEqual({
      trainingRowCount: 100,
      rowCountIsEstimate: true,
    });
  });

  it('scales a capped preview by the bytes actually read', () => {
    expect(estimateTrainingRows([file(1_000, 5_000_000, 500_000)])).toEqual({
      trainingRowCount: 10_000,
      rowCountIsEstimate: true,
    });
  });
});

describe('datasetProblem', () => {
  const read = {
    discoveryError: null,
    hasTraining: true,
    schema: null,
    format: { ok: true, fileErrors: [] },
  };

  it('accepts a dataset whose format was detected', () => {
    expect(datasetProblem({ ...read, schema: INPUTS.dataset.schema })).toBeNull();
  });

  it('says when the files could not be listed', () => {
    expect(datasetProblem({ ...read, discoveryError: new Error('403 Forbidden') })).toBe(
      "Couldn't read the dataset: 403 Forbidden"
    );
  });

  it('says when there is nothing to train on', () => {
    expect(datasetProblem({ ...read, hasTraining: false })).toBe(NO_TRAINING_FILES);
  });

  it('names a file that failed to download instead of blaming the format', () => {
    const fileErrors = [
      { path: 'training/a.jsonl', error: 'Failed to download file: 403 Forbidden' },
      { path: 'training/b.jsonl', error: 'Failed to download file: 403 Forbidden' },
    ];
    expect(datasetProblem({ ...read, format: { ok: false, fileErrors } })).toBe(
      "Couldn't read training/a.jsonl: Failed to download file: 403 Forbidden (1 more file too)"
    );
  });

  it('names a malformed file instead of blaming the format', () => {
    const fileErrors = [{ path: 'training.jsonl', error: 'Line 3 is not valid JSON' }];
    expect(datasetProblem({ ...read, format: { ok: false, fileErrors } })).toBe(
      "Couldn't read training.jsonl: Line 3 is not valid JSON"
    );
  });

  it('blames the format only when every file was read', () => {
    expect(datasetProblem(read)).toBe(UNRECOGNIZED_FORMAT);
  });
});
