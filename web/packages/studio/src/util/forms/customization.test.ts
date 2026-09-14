// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RlGRPOTrainingFinetuningType } from '@nemo/sdk/generated/customizer/schema';
import { customizationJob1, customizationJob3 } from '@studio/mocks/customizer/customization-jobs';
import {
  FORM_DEFAULTS,
  RL_DPO_TRAINING_DEFAULTS,
  RL_GRPO_TRAINING_DEFAULTS,
  customizationFormSchema,
  formToAutomodelCreate,
  formToUnslothCreate,
  getInitialFormValuesFromState,
  jobToFormFields,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import {
  AUTOMODEL_SPEC_DEFAULTS,
  GRPO_SPEC_DEFAULTS,
  numberDefault,
  stringDefault,
} from '@studio/util/forms/specDefaults';

// Deep-clone the defaults so per-test mutations (e.g. flipping finetuning_type)
// never leak through shared nested references into FORM_DEFAULTS or other tests.
/** A fully-valid automodel form value (model + training dataset filled). */
const validAutomodel = (): CustomizationFormFields => {
  const data = structuredClone(FORM_DEFAULTS);
  data.backend = 'automodel';
  data.outputName = 'my-output';
  data.automodel.model = 'default/llama-3.1-8b';
  data.automodel.dataset = { training: 'default/train-ds' };
  return data;
};

/** A fully-valid unsloth form value (model + path filled). */
const validUnsloth = (): CustomizationFormFields => {
  const data = structuredClone(FORM_DEFAULTS);
  data.backend = 'unsloth';
  data.outputName = 'my-output';
  data.unsloth.model.name = 'default/qwen3-1.7b';
  data.unsloth.dataset.path = 'default/train-ds';
  return data;
};

/** A fully-valid GRPO form value (model + dataset + reward environment filled). */
const validGrpo = (): CustomizationFormFields => {
  const data = structuredClone(FORM_DEFAULTS);
  data.backend = 'rl';
  data.outputName = 'my-output';
  data.rl.model = 'default/qwen3-0.6b';
  data.rl.dataset = 'default/gym-data';
  data.rl.training = structuredClone(RL_GRPO_TRAINING_DEFAULTS);
  data.grpo.trainingType = 'grpo';
  data.grpo.environmentFileset = 'default/gym-env';
  return data;
};

const messages = (data: CustomizationFormFields): string[] => {
  const result = customizationFormSchema.safeParse(data);
  return result.success ? [] : result.error.issues.map((i) => i.message);
};

describe('customizationFormSchema', () => {
  it('accepts a valid automodel form', () => {
    expect(customizationFormSchema.safeParse(validAutomodel()).success).toBe(true);
  });

  it('accepts a valid unsloth form', () => {
    expect(customizationFormSchema.safeParse(validUnsloth()).success).toBe(true);
  });

  it('requires an output model name', () => {
    const data = { ...validAutomodel(), outputName: '' };
    expect(messages(data)).toContain('Output model name is required');
  });

  describe('active-backend-only validation', () => {
    // Regression: the form keeps both sub-objects in state; switching backend
    // unmounts the other backend's fields. Only the selected backend must be valid.
    it('ignores an empty automodel subtree when unsloth is selected', () => {
      const data = validUnsloth();
      data.automodel = {
        ...data.automodel,
        model: '',
        dataset: { training: '' },
      };
      expect(customizationFormSchema.safeParse(data).success).toBe(true);
    });

    it('ignores an empty unsloth subtree when automodel is selected', () => {
      const data = validAutomodel();
      data.unsloth = {
        ...data.unsloth,
        model: { ...data.unsloth.model, name: '' },
        dataset: { ...data.unsloth.dataset, path: '' },
      };
      expect(customizationFormSchema.safeParse(data).success).toBe(true);
    });
  });

  describe('automodel required fields', () => {
    it('requires a model', () => {
      const data = validAutomodel();
      data.automodel.model = '';
      expect(messages(data)).toContain('Please select a model');
    });

    it('requires a training dataset', () => {
      const data = validAutomodel();
      data.automodel.dataset.training = '';
      expect(messages(data)).toContain('Training dataset is required');
    });

    it('requires a teacher model for distillation', () => {
      const data = validAutomodel();
      data.automodel.training.training_type = 'distillation';
      data.automodel.training.teacher_model = '';
      expect(messages(data)).toContain('Teacher model is required for distillation');
    });

    it('does not require a teacher model for sft', () => {
      const data = validAutomodel();
      data.automodel.training.training_type = 'sft';
      data.automodel.training.teacher_model = '';
      expect(customizationFormSchema.safeParse(data).success).toBe(true);
    });
  });

  describe('unsloth required fields', () => {
    it('requires a model', () => {
      const data = validUnsloth();
      data.unsloth.model.name = '';
      expect(messages(data)).toContain('Please select a model');
    });

    it('requires a training dataset path', () => {
      const data = validUnsloth();
      data.unsloth.dataset.path = '';
      expect(messages(data)).toContain('Training dataset is required');
    });
  });
});

describe('integrations across backends', () => {
  it('drops an untouched integrations block on every backend', () => {
    const automodel = formToAutomodelCreate(validAutomodel());
    expect(automodel.spec.integrations).toBeUndefined();

    const unsloth = formToUnslothCreate(validUnsloth());
    expect(unsloth.spec.integrations).toBeUndefined();
  });

  it('keeps a configured provider and drops the empty one', () => {
    const data = validAutomodel();
    data.automodel.integrations = { wandb: { project: 'my-project' }, mlflow: {} };

    const { integrations } = formToAutomodelCreate(data).spec;
    expect(integrations?.wandb).toEqual({ project: 'my-project' });
    expect(integrations?.mlflow).toBeUndefined();
  });

  it('treats a blank string as no value rather than an empty name', () => {
    const data = validUnsloth();
    data.unsloth.integrations = { wandb: { project: '   ', entity: 'acme' } };

    const { integrations } = formToUnslothCreate(data).spec;
    expect(integrations?.wandb).toEqual({ entity: 'acme' });
  });
});

describe('formToAutomodelCreate', () => {
  it('maps output name and description onto the job and spec.output', () => {
    const data = validAutomodel();
    data.description = 'my desc';
    const result = formToAutomodelCreate(data);
    expect(result.name).toBe('my-output');
    expect(result.description).toBe('my desc');
    expect(result.spec.output).toEqual({ name: 'my-output', description: 'my desc' });
  });

  it('omits blank job name/description but always sets the required spec.output.name', () => {
    // The top-level job name is omitted when blank, but automodel requires
    // spec.output.name, so it carries the (validation-guaranteed non-empty)
    // output name verbatim.
    const data = validAutomodel();
    data.outputName = '';
    data.description = '';
    const result = formToAutomodelCreate(data);
    expect(result.name).toBeUndefined();
    expect(result.description).toBeUndefined();
    expect(result.spec.output).toEqual({ name: '', description: undefined });
  });

  it('drops the whole distillation block for an sft job', () => {
    const data = validAutomodel();
    data.automodel.training.training_type = 'sft';
    const { training } = formToAutomodelCreate(data).spec;
    expect(training.teacher_model).toBeUndefined();
    expect(training.teacher_precision).toBeUndefined();
    expect(training.distillation_ratio).toBeUndefined();
    expect(training.distillation_temperature).toBeUndefined();
    expect(training.offload_teacher).toBeUndefined();
  });

  it('sends the distillation block for a distillation job', () => {
    const data = validAutomodel();
    data.automodel.training.training_type = 'distillation';
    data.automodel.training.teacher_model = 'default/teacher';
    data.automodel.training.distillation_ratio = 0.7;
    data.automodel.training.distillation_temperature = 2;
    data.automodel.training.teacher_precision = 'fp16';
    data.automodel.training.offload_teacher = true;

    const { training } = formToAutomodelCreate(data).spec;
    expect(training.teacher_model).toBe('default/teacher');
    expect(training.distillation_ratio).toBe(0.7);
    expect(training.distillation_temperature).toBe(2);
    expect(training.teacher_precision).toBe('fp16');
    expect(training.offload_teacher).toBe(true);
  });

  it('keeps lora params for lora finetuning', () => {
    const data = validAutomodel();
    data.automodel.training.finetuning_type = 'lora';
    expect(formToAutomodelCreate(data).spec.training.lora).toBeDefined();
  });

  it('keeps lora params for lora_merged finetuning', () => {
    const data = validAutomodel();
    data.automodel.training.finetuning_type = 'lora_merged';
    expect(formToAutomodelCreate(data).spec.training.lora).toBeDefined();
  });

  it('drops lora params for all_weights finetuning', () => {
    const data = validAutomodel();
    data.automodel.training.finetuning_type = 'all_weights';
    expect(formToAutomodelCreate(data).spec.training.lora).toBeUndefined();
  });

  it('sends the backend-default use_triton flag for lora runs', () => {
    // Backend defaults use_triton to true; the form must not silently send false.
    const spec = formToAutomodelCreate(validAutomodel()).spec;
    expect(spec.training.lora?.use_triton).toBe(true);
  });

  /** Read from the spec, so the test does not go stale the next time a default changes. */
  it('seeds the enum knobs from the spec so the UI matches the backend', () => {
    const spec = formToAutomodelCreate(validAutomodel()).spec;
    expect(spec.training.attn_implementation).toBe(
      stringDefault(AUTOMODEL_SPEC_DEFAULTS, 'training_attn_implementation')
    );
    expect(spec.optimizer?.optimizer).toBe(
      stringDefault(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_optimizer')
    );
    expect(spec.optimizer?.lr_decay_style).toBe(
      stringDefault(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_lr_decay_style')
    );
  });

  it('passes through advanced automodel fields set on the form', () => {
    const data = validAutomodel();
    data.automodel.batch = {
      ...data.automodel.batch!,
      sequence_packing_max_samples: 500,
    };
    data.automodel.training.lora = {
      ...data.automodel.training.lora!,
      exclude_modules: ['*.out_proj'],
    };
    const spec = formToAutomodelCreate(data).spec;
    expect(spec.batch?.sequence_packing_max_samples).toBe(500);
    expect(spec.training.lora?.exclude_modules).toEqual(['*.out_proj']);
  });

  it('includes teacher_model only for distillation', () => {
    const distill = validAutomodel();
    distill.automodel.training.training_type = 'distillation';
    distill.automodel.training.teacher_model = 'default/teacher';
    expect(formToAutomodelCreate(distill).spec.training.teacher_model).toBe('default/teacher');

    const sft = validAutomodel();
    sft.automodel.training.training_type = 'sft';
    sft.automodel.training.teacher_model = 'default/teacher'; // stale value from a prior distillation selection
    expect(formToAutomodelCreate(sft).spec.training.teacher_model).toBeUndefined();
  });
});

describe('formToUnslothCreate', () => {
  it('always supplies the fixed dataset fields the UI does not expose', () => {
    const spec = formToUnslothCreate(validUnsloth()).spec;
    expect(spec.dataset.text_field).toBe('text');
    expect(spec.dataset.packing).toBe(false);
    expect(spec.training?.use_gradient_checkpointing).toBe('unsloth');
  });

  it('preserves the detected apply_chat_template flag', () => {
    const data = validUnsloth();
    data.unsloth.dataset.apply_chat_template = true;
    expect(formToUnslothCreate(data).spec.dataset.apply_chat_template).toBe(true);
  });

  it('omits gpus when blank', () => {
    const data = validUnsloth();
    data.unsloth.hardware = { ...data.unsloth.hardware!, gpus: '' };
    expect(formToUnslothCreate(data).spec.hardware?.gpus).toBeUndefined();
  });

  it('keeps gpus when provided', () => {
    const data = validUnsloth();
    data.unsloth.hardware = { ...data.unsloth.hardware!, gpus: '0,1' };
    expect(formToUnslothCreate(data).spec.hardware?.gpus).toBe('0,1');
  });

  it('keeps lora params for lora finetuning', () => {
    const data = validUnsloth();
    data.unsloth.training!.finetuning_type = 'lora';
    expect(formToUnslothCreate(data).spec.training?.lora).toBeDefined();
  });

  it('drops lora params for all_weights finetuning', () => {
    const data = validUnsloth();
    data.unsloth.training!.finetuning_type = 'all_weights';
    expect(formToUnslothCreate(data).spec.training?.lora).toBeUndefined();
  });

  it('disables quantization for all_weights finetuning', () => {
    // The unsloth backend rejects finetuning_type='all_weights' with 4-bit/8-bit
    // loading, so the mapper must force both off for full-weight runs.
    const data = validUnsloth();
    data.unsloth.training!.finetuning_type = 'all_weights';
    data.unsloth.model.load_in_4bit = true;
    const spec = formToUnslothCreate(data).spec;
    expect(spec.model.load_in_4bit).toBe(false);
    expect(spec.model.load_in_8bit).toBe(false);
  });

  it('keeps quantization for lora finetuning', () => {
    const data = validUnsloth();
    data.unsloth.training!.finetuning_type = 'lora';
    data.unsloth.model.load_in_4bit = true;
    expect(formToUnslothCreate(data).spec.model.load_in_4bit).toBe(true);
  });
});

describe('GRPO defaults', () => {
  const training = RL_GRPO_TRAINING_DEFAULTS as unknown as Record<string, unknown>;

  it('takes every value from the GRPO arm of the spec', () => {
    expect(training.learning_rate).toBe(numberDefault(GRPO_SPEC_DEFAULTS, 'learning_rate'));
    expect(training.adam_eps).toBe(numberDefault(GRPO_SPEC_DEFAULTS, 'adam_eps'));
    expect(training.batch_size).toBe(numberDefault(GRPO_SPEC_DEFAULTS, 'batch_size'));
  });

  it('uses the arm-specific value where DPO and GRPO differ', () => {
    expect(training.ref_policy_kl_penalty).toBe(0);
    expect(
      (RL_DPO_TRAINING_DEFAULTS as unknown as Record<string, unknown>).ref_policy_kl_penalty
    ).toBe(0.05);
  });

  it.each(['val_check_interval', 'max_steps', 'seed', 'min_learning_rate'])(
    'leaves %s unset because the spec declares no default',
    (field) => {
      expect(training[field]).toBeUndefined();
    }
  );
});

describe('GRPO form validation', () => {
  it('accepts the default GRPO form', () => {
    expect(messages(validGrpo())).toEqual([]);
  });

  it('requires a reward environment fileset', () => {
    const data = validGrpo();
    data.grpo.environmentFileset = '';
    expect(messages(data)).toContain('A reward environment fileset is required for GRPO training');
  });

  it('rejects max new tokens above the max sequence length', () => {
    const data = validGrpo();
    data.rl.training.max_seq_length = 2048;
    data.grpo.max_new_tokens = 4096;
    expect(messages(data).join(' ')).toContain('cannot exceed the max sequence length');
  });

  // The backend only catches this after the job is accepted and scheduled onto GPUs.
  it('rejects a rollout batch that is not a multiple of the global batch size', () => {
    const data = validGrpo();
    data.rl.training.batch_size = 32;
    data.grpo.num_prompts_per_step = 9;
    data.grpo.num_generations_per_prompt = 8;
    expect(messages(data).join(' ')).toContain('must be a multiple of the global batch size (32)');
  });

  it('accepts a rollout batch that is an exact multiple', () => {
    const data = validGrpo();
    data.rl.training.batch_size = 32;
    data.grpo.num_prompts_per_step = 8;
    data.grpo.num_generations_per_prompt = 8;
    expect(messages(data)).toEqual([]);
  });

  // _build_lora_cfg turns use_triton off itself when tp > 1, so blocking the
  // combination here would refuse a job the backend runs fine.
  it.each([1, 2])('accepts Triton LoRA kernels at tensor parallel size %i', (tp) => {
    const data = validGrpo();
    data.grpo.finetuning_type = RlGRPOTrainingFinetuningType.lora;
    data.grpo.lora = { ...data.grpo.lora, use_triton: true };
    data.rl.training.parallelism = { ...data.rl.training.parallelism, tensor_parallel_size: tp };
    expect(messages(data)).toEqual([]);
  });

  // Clearing the input leaves the field undefined and the backend derives it, so the
  // rule still has to be checked: 32 / 5 floors to 6, giving 30 against a batch of 32.
  it('rejects a derived rollout batch that is not a multiple of the batch size', () => {
    const data = validGrpo();
    data.rl.training.batch_size = 32;
    data.grpo.num_generations_per_prompt = 5;
    data.grpo.num_prompts_per_step = undefined as unknown as number;
    expect(messages(data).join(' ')).toContain('must be a multiple of the global batch size (32)');
  });
});

describe('jobToFormFields', () => {
  it('maps an automodel job onto the automodel backend', () => {
    const fields = jobToFormFields(customizationJob1);
    expect(fields.backend).toBe('automodel');
    expect(fields.automodel.model).toBe(customizationJob1.spec.model);
    expect(fields.description).toBe(customizationJob1.description);
  });

  it('maps an unsloth job onto the unsloth backend', () => {
    const fields = jobToFormFields(customizationJob3);
    expect(fields.backend).toBe('unsloth');
    expect(fields.unsloth.model.name).toBe(customizationJob3.spec.model.name);
  });

  it('generates a fresh output name rather than reusing the source job name', () => {
    const fields = jobToFormFields(customizationJob1);
    expect(fields.outputName).toBeTruthy();
    expect(fields.outputName).not.toBe(customizationJob1.name);
  });

  it('strips nulls out of the stored spec so optional fields fall back to undefined', () => {
    const jobWithNulls = {
      ...customizationJob1,
      spec: { ...customizationJob1.spec, optimizer: { learning_rate: null } },
    } as unknown as typeof customizationJob1;
    const fields = jobToFormFields(jobWithNulls);
    expect(fields.automodel.optimizer?.learning_rate).toBeUndefined();
  });
});

describe('getInitialFormValuesFromState', () => {
  it('returns template initialValues as-is when the backend is valid', () => {
    const initialValues = validAutomodel();
    expect(getInitialFormValuesFromState({ initialValues })).toBe(initialValues);
  });

  it('converts a cloneFromJob into form fields', () => {
    const fields = getInitialFormValuesFromState({ cloneFromJob: customizationJob1 });
    expect(fields?.backend).toBe('automodel');
    expect(fields?.automodel.model).toBe(customizationJob1.spec.model);
  });

  it('prefers template initialValues over cloneFromJob when both are present', () => {
    const initialValues = validUnsloth();
    const fields = getInitialFormValuesFromState({
      initialValues,
      cloneFromJob: customizationJob1,
    });
    expect(fields).toBe(initialValues);
  });

  it('ignores initialValues with an unrecognized backend', () => {
    expect(getInitialFormValuesFromState({ initialValues: { backend: 'nope' } })).toBeUndefined();
  });

  it('ignores initialValues with a valid backend but no matching spec', () => {
    expect(
      getInitialFormValuesFromState({ initialValues: { backend: 'automodel' } })
    ).toBeUndefined();
    expect(
      getInitialFormValuesFromState({ initialValues: { backend: 'unsloth' } })
    ).toBeUndefined();
  });

  it('ignores a cloneFromJob whose spec matches no backend', () => {
    expect(
      getInitialFormValuesFromState({ cloneFromJob: { spec: { foo: 'bar' } } })
    ).toBeUndefined();
  });

  it.each([undefined, null, 'string', 42, {}, { other: 1 }])(
    'returns undefined for unrelated state (%s)',
    (state) => {
      expect(getInitialFormValuesFromState(state)).toBeUndefined();
    }
  );
});
