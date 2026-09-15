// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { RadioCard } from '@nemo/common/src/components/RadioCard';
import {
  AutomodelTrainingSpecFinetuningType,
  AutomodelTrainingSpecTrainingType,
  UnslothTrainingSpecFinetuningType,
} from '@nemo/sdk/generated/customizer/schema';
import { RadioGroupRoot, Stack, Text } from '@nvidia/foundations-react-core';
import { TEACHER_PRECISION_ITEMS } from '@studio/components/NewCustomizationForm/constants';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import {
  RL_DPO_TRAINING_DEFAULTS,
  RL_GRPO_TRAINING_DEFAULTS,
  type CustomizationFormFields,
} from '@studio/util/forms/customization';
import { AUTOMODEL_SPEC_DEFAULTS, specSliderProps } from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

const AUTOMODEL_FINETUNING_TYPES = [
  {
    value: AutomodelTrainingSpecFinetuningType.lora,
    title: 'LoRA',
    description: 'Low-rank adapter — fewer parameters, less VRAM.',
  },
  {
    value: AutomodelTrainingSpecFinetuningType.lora_merged,
    title: 'LoRA (Merged)',
    description: 'LoRA weights merged into base at the end.',
  },
  {
    value: AutomodelTrainingSpecFinetuningType.all_weights,
    title: 'Full Weights',
    description: 'Train all model parameters.',
  },
] as const;

const UNSLOTH_FINETUNING_TYPES = [
  {
    value: UnslothTrainingSpecFinetuningType.lora,
    title: 'LoRA',
    description: 'Low-rank adapter — fewer parameters, less VRAM.',
  },
  {
    value: UnslothTrainingSpecFinetuningType.all_weights,
    title: 'Full Weights',
    description: 'Train all model parameters.',
  },
] as const;

/**
 * Bound to `grpo.trainingType` rather than `rl.training.type`: the form holds one
 * `rl.training` object, and flipping the union discriminator in place would leave it
 * carrying the other arm's fields. `formToRlCreate` sets `type` from this on submit.
 */
const RL_TRAINING_TYPES = [
  {
    value: 'dpo',
    title: 'DPO',
    description: 'Direct Preference Optimization — full-weight fine-tuning on preference pairs.',
  },
  {
    value: 'grpo',
    title: 'GRPO',
    description: 'Group Relative Policy Optimization — RL fine-tuning with a reward environment.',
  },
] as const;

const AUTOMODEL_TRAINING_TYPES = [
  {
    value: AutomodelTrainingSpecTrainingType.sft,
    title: 'SFT',
    description: 'Supervised fine-tuning on instruction/response pairs.',
  },
  {
    value: AutomodelTrainingSpecTrainingType.distillation,
    title: 'Distillation',
    description: 'Learn from a larger teacher model.',
  },
] as const;

export const TrainingMethodSection = () => {
  const { watch, setValue, getValues, control, formState } =
    useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  if (backend === 'rl') {
    const rlTrainingType = watch('grpo.trainingType');
    return (
      <FormSection title="Training Method">
        <RadioGroupRoot
          name="rlTrainingType"
          value={rlTrainingType ?? 'dpo'}
          onValueChange={(v) => {
            const next = v as 'dpo' | 'grpo';
            setValue('grpo.trainingType', next, { shouldValidate: true });
            // rl.training is shared, so reset to the incoming method's defaults, then
            // carry over the fields that mean the same thing under either method.
            const current = getValues('rl.training');
            setValue(
              'rl.training',
              {
                ...(next === 'grpo' ? RL_GRPO_TRAINING_DEFAULTS : RL_DPO_TRAINING_DEFAULTS),
                parallelism: current.parallelism,
                batch_size: current.batch_size,
                max_seq_length: current.max_seq_length,
                // learning_rate is NOT carried over: the two methods now default two
                // orders of magnitude apart, so each switch takes its own default.
              },
              { shouldValidate: true }
            );
          }}
          className="w-full"
          disabled={disabled}
        >
          <div className="grid grid-cols-2 gap-4">
            {RL_TRAINING_TYPES.map((opt) => (
              <RadioCard
                key={opt.value}
                value={opt.value}
                label={<Text kind="body/bold/md">{opt.title}</Text>}
                description={
                  <Text kind="body/regular/md" color="secondary">
                    {opt.description}
                  </Text>
                }
                labelSide="left"
              />
            ))}
          </div>
        </RadioGroupRoot>
      </FormSection>
    );
  }

  const finetuningType =
    backend === 'automodel'
      ? watch('automodel.training.finetuning_type')
      : watch('unsloth.training.finetuning_type');

  const trainingType = backend === 'automodel' ? watch('automodel.training.training_type') : null;

  const finetuningOptions =
    backend === 'automodel' ? AUTOMODEL_FINETUNING_TYPES : UNSLOTH_FINETUNING_TYPES;

  const handleFinetuningChange = (value: string) => {
    if (backend === 'automodel') {
      setValue(
        'automodel.training.finetuning_type',
        value as CustomizationFormFields['automodel']['training']['finetuning_type'],
        { shouldValidate: true }
      );
    } else {
      setValue('unsloth.training.finetuning_type', value as UnslothTrainingSpecFinetuningType, {
        shouldValidate: true,
      });
    }
  };

  return (
    <FormSection title="Training Method">
      <Stack gap="density-xl">
        <Stack gap="density-md">
          <Text kind="label/bold/md">Fine-tuning Type</Text>
          <RadioGroupRoot
            name="finetuningType"
            value={finetuningType ?? ''}
            onValueChange={handleFinetuningChange}
            className="w-full"
            disabled={disabled}
          >
            <div className="grid grid-cols-2 gap-4">
              {finetuningOptions.map((opt) => (
                <RadioCard
                  key={opt.value}
                  value={opt.value}
                  label={<Text kind="body/bold/md">{opt.title}</Text>}
                  description={
                    <Text kind="body/regular/md" color="secondary">
                      {opt.description}
                    </Text>
                  }
                  labelSide="left"
                />
              ))}
            </div>
          </RadioGroupRoot>
        </Stack>

        {backend === 'automodel' && (
          <Stack gap="density-md">
            <Text kind="label/bold/md">Training Type</Text>
            <RadioGroupRoot
              name="trainingType"
              value={trainingType ?? 'sft'}
              onValueChange={(v) =>
                setValue(
                  'automodel.training.training_type',
                  v as AutomodelTrainingSpecTrainingType,
                  {
                    shouldValidate: true,
                  }
                )
              }
              className="w-full"
              disabled={disabled}
            >
              <div className="grid grid-cols-2 gap-4">
                {AUTOMODEL_TRAINING_TYPES.map((opt) => (
                  <RadioCard
                    key={opt.value}
                    value={opt.value}
                    label={<Text kind="body/bold/md">{opt.title}</Text>}
                    description={
                      <Text kind="body/regular/md" color="secondary">
                        {opt.description}
                      </Text>
                    }
                    labelSide="left"
                  />
                ))}
              </div>
            </RadioGroupRoot>
            {trainingType === 'distillation' && (
              <Stack gap="density-lg">
                <ControlledTextInput
                  useControllerProps={{ name: 'automodel.training.teacher_model', control }}
                  label="Teacher Model"
                  placeholder="workspace/model-name"
                  required
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'automodel.training.distillation_ratio', control }}
                  formFieldProps={{
                    slotLabel: 'Distillation Ratio',
                    slotInfo:
                      'How much of the loss comes from matching the teacher rather than the training labels. 0 is pure supervised training, 1 is pure distillation.',
                  }}
                  {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'training_distillation_ratio')}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{
                    name: 'automodel.training.distillation_temperature',
                    control,
                  }}
                  formFieldProps={{
                    slotLabel: 'Distillation Temperature',
                    slotInfo:
                      'Softens the teacher’s output distribution. Higher values spread probability mass onto lower-ranked tokens, so the student learns more than just the teacher’s top choice.',
                  }}
                  {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'training_distillation_temperature')}
                  min={0.1}
                  max={10}
                  step={0.1}
                  disabled={disabled}
                />
                <ControlledSelect
                  useControllerProps={{ name: 'automodel.training.teacher_precision', control }}
                  formFieldProps={{
                    slotLabel: 'Teacher Precision',
                    slotInfo: 'Precision the teacher runs at. Lower precision frees VRAM.',
                  }}
                  items={TEACHER_PRECISION_ITEMS}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'automodel.training.offload_teacher', control }}
                  formFieldProps={{
                    slotLabel: 'Offload Teacher',
                    labelPosition: 'left',
                    slotInfo:
                      'Keep the teacher on CPU between forward passes. Frees VRAM for the student at the cost of transfer time.',
                  }}
                  disabled={disabled}
                />
              </Stack>
            )}
          </Stack>
        )}
      </Stack>
    </FormSection>
  );
};
