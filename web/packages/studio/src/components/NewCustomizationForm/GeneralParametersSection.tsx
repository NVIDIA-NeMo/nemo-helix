// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  AUTOMODEL_PRECISION_ITEMS,
  AUTOMODEL_RECIPE_ITEMS,
  OPTIMIZER_TYPE_ITEMS,
  UNSLOTH_GRADIENT_CHECKPOINTING_ITEMS,
  UNSLOTH_LR_SCHEDULER_ITEMS,
  UNSLOTH_OPTIM_ITEMS,
  UNSLOTH_PRECISION_ITEMS,
} from '@studio/components/NewCustomizationForm/constants';
import { ControlledJsonInput } from '@studio/components/NewCustomizationForm/ControlledJsonInput';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import { ProgressReportingFields } from '@studio/components/NewCustomizationForm/ProgressReportingFields';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import {
  AUTOMODEL_SPEC_DEFAULTS,
  DPO_SPEC_DEFAULTS,
  UNSLOTH_SPEC_DEFAULTS,
  specSliderProps,
} from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

export const GeneralParametersSection = () => {
  const { control, watch, setValue, formState } = useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  if (backend === 'rl') {
    return (
      <FormSection title="Training Parameters">
        <Stack gap="density-lg">
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'rl.training.epochs', control }}
            formFieldProps={{
              slotLabel: 'Epochs',
              slotInfo: 'Number of complete passes through the dataset.',
            }}
            {...specSliderProps(DPO_SPEC_DEFAULTS, 'epochs')}
            min={1}
            max={100}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'rl.training.learning_rate', control }}
            formFieldProps={{
              slotLabel: 'Learning Rate',
              slotInfo: 'Learning rate reached at the end of warmup, before decay begins.',
            }}
            {...specSliderProps(DPO_SPEC_DEFAULTS, 'learning_rate')}
            min={1e-6}
            max={1e-3}
            step={1e-6}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'rl.training.batch_size', control }}
            formFieldProps={{
              slotLabel: 'Global Batch Size',
              slotInfo: 'Examples per optimizer step, summed across all GPUs.',
            }}
            {...specSliderProps(DPO_SPEC_DEFAULTS, 'batch_size')}
            min={1}
            max={256}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'rl.training.max_seq_length', control }}
            formFieldProps={{
              slotLabel: 'Max Sequence Length',
              slotInfo:
                'Maximum token sequence length for training; longer sequences are truncated.',
            }}
            {...specSliderProps(DPO_SPEC_DEFAULTS, 'max_seq_length')}
            min={128}
            max={131072}
            step={128}
            disabled={disabled}
          />
          <AccordionRoot multiple>
            <AccordionItem value="advanced" className="border-b-0">
              <AccordionTrigger>
                <Text kind="label/bold/md">Advanced</Text>
              </AccordionTrigger>
              <AccordionContent>
                <Stack gap="density-md" className="pt-density-md">
                  {/*
                    No Max Steps control here on purpose. DPO run length is driven by
                    Epochs, matching the backend default (`max_steps: None`), so
                    `RL_DPO_TRAINING_DEFAULTS` leaves it unset. A slider must declare a
                    reset target, and any number it resets to would silently switch the
                    run to a step budget — max_steps overrides epochs. GRPO exposes it
                    because GRPO genuinely defaults to a step budget.
                  */}
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.micro_batch_size', control }}
                    formFieldProps={{
                      slotLabel: 'Micro Batch Size',
                      slotInfo:
                        'Examples each GPU processes at once. Lower this first if training runs out of memory.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'micro_batch_size')}
                    min={1}
                    max={64}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.warmup_steps', control }}
                    formFieldProps={{
                      slotLabel: 'Warmup Steps',
                      slotInfo:
                        'Steps spent ramping the learning rate up from zero. Around 10% of total steps is a stable starting point.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'warmup_steps')}
                    min={0}
                    max={1000}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.weight_decay', control }}
                    formFieldProps={{
                      slotLabel: 'Weight Decay',
                      slotInfo: 'Penalty on large weights. Higher regularizes more; 0 disables it.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'weight_decay')}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.seed', control }}
                    formFieldProps={{
                      slotLabel: 'Seed',
                      slotInfo: 'Random seed for reproducibility.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'seed')}
                    min={0}
                    max={999999}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSwitch
                    useControllerProps={{ name: 'rl.training.activation_checkpointing', control }}
                    formFieldProps={{
                      slotLabel: 'Activation Checkpoints',
                      labelPosition: 'left',
                      slotInfo:
                        'Recompute activations during the backward pass to reduce memory at the cost of compute.',
                    }}
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'rl.training.optimizer_type', control }}
                    formFieldProps={{
                      slotLabel: 'Optimizer Type',
                      slotInfo:
                        'Optimizer + LR-scheduler combination (AdamW/Adam × cosine-annealing/flat-LR). Defaults to AdamW with cosine annealing.',
                    }}
                    items={OPTIMIZER_TYPE_ITEMS}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.min_learning_rate', control }}
                    formFieldProps={{
                      slotLabel: 'Min Learning Rate',
                      slotInfo:
                        'Floor the cosine schedule decays to. Keep it below the peak learning rate.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'min_learning_rate')}
                    min={0}
                    max={1e-3}
                    step={1e-6}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.adam_beta1', control }}
                    formFieldProps={{
                      slotLabel: 'Adam β₁',
                      slotInfo: 'Decay rate for the gradient moving average (momentum).',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'adam_beta1')}
                    min={0}
                    max={0.999}
                    step={0.001}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.adam_beta2', control }}
                    formFieldProps={{
                      slotLabel: 'Adam β₂',
                      slotInfo: 'Decay rate for the squared-gradient moving average.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'adam_beta2')}
                    min={0}
                    max={0.9999}
                    step={0.0001}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.adam_eps', control }}
                    formFieldProps={{
                      slotLabel: 'Adam ε',
                      slotInfo: 'Numerical stability term.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'adam_eps')}
                    min={1e-10}
                    max={1e-6}
                    step={1e-10}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.val_check_interval', control }}
                    formFieldProps={{
                      slotLabel: 'Val Check Interval',
                      slotInfo:
                        'Validation frequency. Values ≤ 1.0 are a fraction of an epoch; values > 1.0 are a step count.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'val_check_interval')}
                    min={0.01}
                    max={1000}
                    step={0.01}
                    disabled={disabled}
                  />
                  <ProgressReportingFields
                    prefix="rl.training"
                    defaults={DPO_SPEC_DEFAULTS}
                    defaultsPrefix="progress_reporting"
                    disabled={disabled}
                  />
                  <ControlledSwitch
                    useControllerProps={{ name: 'rl.training.val_at_end', control }}
                    formFieldProps={{
                      slotLabel: 'Validate at End',
                      labelPosition: 'left',
                      slotInfo: 'Run a final validation pass after the last training step.',
                    }}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'rl.training.keep_top_k', control }}
                    formFieldProps={{
                      slotLabel: 'Keep Top-K Checkpoints',
                      slotInfo: 'Number of best checkpoints to retain, ranked by validation loss.',
                    }}
                    {...specSliderProps(DPO_SPEC_DEFAULTS, 'keep_top_k')}
                    min={1}
                    max={10}
                    step={1}
                    disabled={disabled}
                  />
                </Stack>
              </AccordionContent>
            </AccordionItem>
          </AccordionRoot>
        </Stack>
      </FormSection>
    );
  }

  if (backend === 'automodel') {
    return (
      <FormSection title="Training Parameters">
        <Stack gap="density-lg">
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.schedule.epochs', control }}
            formFieldProps={{
              slotLabel: 'Epochs',
              slotInfo: 'Number of complete passes through the dataset.',
            }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'schedule_epochs')}
            min={1}
            max={100}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.optimizer.learning_rate', control }}
            formFieldProps={{
              slotLabel: 'Learning Rate',
              slotInfo: 'Peak learning rate. Start near 5e-5 for full SFT, or 1e-4 with LoRA.',
            }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_learning_rate')}
            min={1e-6}
            max={1e-3}
            step={1e-6}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.batch.global_batch_size', control }}
            formFieldProps={{
              slotLabel: 'Global Batch Size',
              slotInfo: 'Examples per optimizer step, summed across all GPUs.',
            }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'batch_global_batch_size')}
            min={1}
            max={256}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.training.max_seq_length', control }}
            formFieldProps={{
              slotLabel: 'Max Sequence Length',
              slotInfo:
                'Maximum token sequence length for training; longer sequences are truncated.',
            }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'training_max_seq_length')}
            min={128}
            max={131072}
            step={128}
            disabled={disabled}
          />
          <ControlledSwitch
            useControllerProps={{ name: 'automodel.batch.sequence_packing', control }}
            formFieldProps={{
              slotLabel: 'Sequence Packing',
              slotInfo: 'Packs short examples together to improve training speed.',
              labelPosition: 'left',
            }}
            disabled={disabled}
          />
          <AccordionRoot multiple>
            <AccordionItem value="advanced" className="border-b-0">
              <AccordionTrigger>
                <Text kind="label/bold/md">Advanced</Text>
              </AccordionTrigger>
              <AccordionContent>
                <Stack gap="density-md" className="pt-density-md">
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.batch.micro_batch_size', control }}
                    formFieldProps={{
                      slotLabel: 'Micro Batch Size',
                      slotInfo:
                        'Examples each GPU processes at once. Lower this first if training runs out of memory.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'batch_micro_batch_size')}
                    min={1}
                    max={64}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.schedule.max_steps', control }}
                    formFieldProps={{
                      slotLabel: 'Max Steps',
                      slotInfo:
                        'Hard cap on training steps. Training stops at whichever comes first, this or the epoch count.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'schedule_max_steps')}
                    min={1}
                    max={100000}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.schedule.val_check_interval', control }}
                    formFieldProps={{
                      slotLabel: 'Val Check Interval',
                      slotInfo:
                        'Validation frequency. Values ≤ 1.0 are a fraction of an epoch; values > 1.0 are a step count.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'schedule_val_check_interval')}
                    min={0.01}
                    max={1000}
                    step={0.01}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.schedule.seed', control }}
                    formFieldProps={{
                      slotLabel: 'Seed',
                      slotInfo: 'Random seed for reproducibility.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'schedule_seed')}
                    min={0}
                    max={999999}
                    step={1}
                    disabled={disabled}
                  />
                  <ProgressReportingFields
                    prefix="automodel.schedule"
                    defaults={AUTOMODEL_SPEC_DEFAULTS}
                    defaultsPrefix="schedule_progress_reporting"
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.warmup_steps', control }}
                    formFieldProps={{
                      slotLabel: 'Warmup Steps',
                      slotInfo:
                        'Steps spent ramping the learning rate up from zero. Around 10% of total steps is a stable starting point.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_warmup_steps')}
                    min={0}
                    max={1000}
                    step={1}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.weight_decay', control }}
                    formFieldProps={{
                      slotLabel: 'Weight Decay',
                      slotInfo: 'Penalty on large weights. Higher regularizes more; 0 disables it.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_weight_decay')}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.min_learning_rate', control }}
                    formFieldProps={{
                      slotLabel: 'Min Learning Rate',
                      slotInfo:
                        'Floor the cosine schedule decays to. Keep it below the peak learning rate.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_min_learning_rate')}
                    min={0}
                    max={1e-3}
                    step={1e-6}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.adam_eps', control }}
                    formFieldProps={{
                      slotLabel: 'Adam Epsilon',
                      slotInfo: 'Adam/AdamW epsilon for numerical stability.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_adam_eps')}
                    min={1e-10}
                    max={1e-6}
                    step={1e-10}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.adam_beta1', control }}
                    formFieldProps={{
                      slotLabel: 'Adam β₁',
                      slotInfo: 'Decay rate for the gradient moving average (momentum).',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_adam_beta1')}
                    min={0}
                    max={0.999}
                    step={0.001}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{ name: 'automodel.optimizer.adam_beta2', control }}
                    formFieldProps={{
                      slotLabel: 'Adam β₂',
                      slotInfo: 'Decay rate for the squared-gradient moving average.',
                    }}
                    {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'optimizer_adam_beta2')}
                    min={0}
                    max={0.9999}
                    step={0.0001}
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'automodel.optimizer.optimizer', control }}
                    formFieldProps={{
                      slotLabel: 'Optimizer',
                      slotInfo:
                        "Optimizer algorithm. 'auto' selects Transformer Engine FusedAdam for retrieval recipes and torch Adam for SFT.",
                    }}
                    items={[
                      { value: 'Adam', children: 'Adam' },
                      { value: 'AdamW', children: 'AdamW' },
                    ]}
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'automodel.optimizer.lr_decay_style', control }}
                    formFieldProps={{
                      slotLabel: 'LR Decay Style',
                      slotInfo: 'Learning-rate decay schedule.',
                    }}
                    items={[
                      { value: 'cosine', children: 'Cosine' },
                      { value: 'linear', children: 'Linear' },
                      { value: 'constant', children: 'Constant' },
                    ]}
                    disabled={disabled}
                  />
                  <ControlledTextInput
                    useControllerProps={{ name: 'automodel.dataset.prompt_template', control }}
                    formFieldProps={{
                      slotLabel: 'Prompt Template',
                      slotInfo:
                        'Only needed when your columns are neither messages nor prompt/completion. Use exactly two placeholders naming your columns.',
                    }}
                    placeholder="{input} {output}"
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'automodel.training.precision', control }}
                    formFieldProps={{
                      slotLabel: 'Precision',
                      slotInfo: 'Leave unset to let the backend choose for the hardware.',
                    }}
                    items={AUTOMODEL_PRECISION_ITEMS}
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'automodel.training.recipe', control }}
                    formFieldProps={{
                      slotLabel: 'Recipe',
                      slotInfo: 'Auto picks the recipe from the model architecture.',
                    }}
                    items={AUTOMODEL_RECIPE_ITEMS}
                    disabled={disabled}
                  />
                  <ControlledTextInput
                    useControllerProps={{ name: 'automodel.training.execution_profile', control }}
                    formFieldProps={{
                      slotLabel: 'Execution Profile',
                      slotInfo:
                        'Operator-configured GPU profile for the training step, e.g. h100. Leave blank to use the service default.',
                    }}
                    placeholder="Service default"
                    disabled={disabled}
                  />
                  <ControlledSelect
                    useControllerProps={{ name: 'automodel.training.attn_implementation', control }}
                    formFieldProps={{
                      slotLabel: 'Attention Implementation',
                      slotInfo:
                        "Attention backend: 'sdpa' (PyTorch native), 'flash_attention_2', or 'eager'.",
                    }}
                    items={[
                      { value: 'sdpa', children: 'SDPA' },
                      { value: 'flash_attention_2', children: 'FlashAttention 2' },
                      { value: 'eager', children: 'Eager' },
                    ]}
                    disabled={disabled}
                  />
                  <ControlledSliderWithTextInput
                    useControllerProps={{
                      name: 'automodel.batch.sequence_packing_max_samples',
                      control,
                    }}
                    formFieldProps={{
                      slotLabel: 'Sequence Packing Max Samples',
                      slotInfo:
                        'Samples analyzed to estimate the optimal pack size when packing is enabled.',
                    }}
                    {...specSliderProps(
                      AUTOMODEL_SPEC_DEFAULTS,
                      'batch_sequence_packing_max_samples'
                    )}
                    min={1}
                    max={10000}
                    step={1}
                    disabled={disabled}
                  />
                </Stack>
              </AccordionContent>
            </AccordionItem>
          </AccordionRoot>
        </Stack>
      </FormSection>
    );
  }

  return (
    <FormSection title="Training Parameters">
      <Stack gap="density-lg">
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.schedule.epochs', control }}
          formFieldProps={{
            slotLabel: 'Epochs',
            slotInfo: 'Number of complete passes through the dataset.',
          }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_epochs')}
          min={1}
          max={100}
          step={1}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.optimizer.learning_rate', control }}
          formFieldProps={{
            slotLabel: 'Learning Rate',
            slotInfo: 'Learning rate reached at the end of warmup, before decay begins.',
          }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_learning_rate')}
          min={1e-6}
          max={1e-3}
          step={1e-6}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.batch.per_device_train_batch_size', control }}
          formFieldProps={{
            slotLabel: 'Per-Device Batch Size',
            slotInfo:
              'Examples per GPU per step. Keep small for large models to avoid running out of memory.',
          }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'batch_per_device_train_batch_size')}
          min={1}
          max={64}
          step={1}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.model.max_seq_length', control }}
          formFieldProps={{
            slotLabel: 'Max Sequence Length',
            slotInfo: 'Maximum token sequence length for training; longer sequences are truncated.',
          }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'model_max_seq_length')}
          min={128}
          max={131072}
          step={128}
          disabled={disabled}
        />
        <AccordionRoot multiple>
          <AccordionItem value="advanced">
            <AccordionTrigger>
              <Text kind="label/bold/md">Advanced</Text>
            </AccordionTrigger>
            <AccordionContent>
              <Stack gap="density-md" className="pt-density-md">
                <ControlledSliderWithTextInput
                  useControllerProps={{
                    name: 'unsloth.batch.gradient_accumulation_steps',
                    control,
                  }}
                  formFieldProps={{
                    slotLabel: 'Gradient Accumulation Steps',
                    slotInfo:
                      'Steps accumulated before an optimizer update. Raises the effective batch size without using more memory.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'batch_gradient_accumulation_steps')}
                  min={1}
                  max={64}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.max_steps', control }}
                  formFieldProps={{
                    slotLabel: 'Max Steps',
                    slotInfo:
                      'Hard cap on training steps. Training stops at whichever comes first, this or the epoch count.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_max_steps')}
                  min={1}
                  max={100000}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.warmup_ratio', control }}
                  formFieldProps={{
                    slotLabel: 'Warmup Ratio',
                    slotInfo:
                      'Warmup length as a fraction of total steps. An alternative to Warmup Steps.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_warmup_ratio')}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.eval_steps', control }}
                  formFieldProps={{
                    slotLabel: 'Eval Steps',
                    slotInfo: 'Run evaluation every N steps.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_eval_steps')}
                  min={1}
                  max={10000}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.save_steps', control }}
                  formFieldProps={{
                    slotLabel: 'Save Steps',
                    slotInfo: 'Write a checkpoint every N steps.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_save_steps')}
                  min={1}
                  max={10000}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.logging_steps', control }}
                  formFieldProps={{
                    slotLabel: 'Logging Steps',
                    slotInfo: 'Emit training metrics every N steps.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_logging_steps')}
                  min={1}
                  max={1000}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.seed', control }}
                  formFieldProps={{
                    slotLabel: 'Seed',
                    slotInfo: 'Random seed for reproducibility.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_seed')}
                  min={0}
                  max={999999}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSelect
                  useControllerProps={{ name: 'unsloth.schedule.lr_scheduler_type', control }}
                  formFieldProps={{
                    slotLabel: 'LR Scheduler',
                    slotInfo: 'Shape of the learning-rate decay after warmup.',
                  }}
                  items={UNSLOTH_LR_SCHEDULER_ITEMS}
                  disabled={disabled}
                />
                <ControlledSelect
                  useControllerProps={{ name: 'unsloth.optimizer.optim', control }}
                  formFieldProps={{
                    slotLabel: 'Optimizer',
                    slotInfo: 'The 8-bit variants trade a little precision for much less VRAM.',
                  }}
                  items={UNSLOTH_OPTIM_ITEMS}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.schedule.warmup_steps', control }}
                  formFieldProps={{
                    slotLabel: 'Warmup Steps',
                    slotInfo:
                      'Steps spent ramping the learning rate up from zero. Around 10% of total steps is a stable starting point.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'schedule_warmup_steps')}
                  min={0}
                  max={1000}
                  step={1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.weight_decay', control }}
                  formFieldProps={{
                    slotLabel: 'Weight Decay',
                    slotInfo: 'Penalty on large weights. Higher regularizes more; 0 disables it.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_weight_decay')}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.adam_beta1', control }}
                  formFieldProps={{
                    slotLabel: 'Adam β₁',
                    slotInfo: 'Decay rate for the gradient moving average (momentum).',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_adam_beta1')}
                  min={0}
                  max={1}
                  step={0.001}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.adam_beta2', control }}
                  formFieldProps={{
                    slotLabel: 'Adam β₂',
                    slotInfo: 'Decay rate for the squared-gradient moving average.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_adam_beta2')}
                  min={0}
                  max={1}
                  step={0.001}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.adam_epsilon', control }}
                  formFieldProps={{
                    slotLabel: 'Adam Epsilon',
                    slotInfo: 'Adam/AdamW epsilon for numerical stability.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_adam_epsilon')}
                  min={1e-10}
                  max={1e-6}
                  step={1e-10}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.max_grad_norm', control }}
                  formFieldProps={{
                    slotLabel: 'Max Gradient Norm',
                    slotInfo: 'Gradient-clipping max norm (TRL default 1.0).',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_max_grad_norm')}
                  min={0}
                  max={10}
                  step={0.1}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.label_smoothing_factor', control }}
                  formFieldProps={{
                    slotLabel: 'Label Smoothing Factor',
                    slotInfo: 'Label smoothing for the cross-entropy loss. 0.0 disables.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_label_smoothing_factor')}
                  min={0}
                  max={1}
                  step={0.01}
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.optimizer.neftune_noise_alpha', control }}
                  formFieldProps={{
                    slotLabel: 'NEFTune Noise Alpha',
                    slotInfo:
                      'NEFTune embedding-noise alpha (quality boost). Leave blank to disable.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'optimizer_neftune_noise_alpha')}
                  min={0}
                  max={20}
                  step={1}
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{ name: 'unsloth.schedule.lr_scheduler_kwargs', control }}
                  formFieldProps={{
                    slotLabel: 'LR Scheduler Kwargs (JSON)',
                    slotInfo:
                      'Extra kwargs for the LR scheduler, e.g. {"num_cycles": 3} for cosine_with_restarts. Leave blank to use the scheduler defaults.',
                  }}
                  placeholder='{ "num_cycles": 1 }'
                  disabled={disabled}
                />
                <ProgressReportingFields
                  prefix="unsloth.schedule"
                  defaults={UNSLOTH_SPEC_DEFAULTS}
                  defaultsPrefix="schedule_progress_reporting"
                  disabled={disabled}
                />
                <ControlledTextInput
                  useControllerProps={{ name: 'unsloth.dataset.text_field', control }}
                  formFieldProps={{
                    slotLabel: 'Text Field',
                    slotInfo: 'Column in the dataset holding the training text.',
                  }}
                  placeholder="text"
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.dataset.packing', control }}
                  formFieldProps={{
                    slotLabel: 'Pack Sequences',
                    labelPosition: 'left',
                    slotInfo:
                      'Concatenate short examples up to the max sequence length. Better throughput, at the cost of examples spanning a boundary.',
                  }}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.dataset.apply_chat_template', control }}
                  formFieldProps={{
                    slotLabel: 'Apply Chat Template',
                    labelPosition: 'left',
                    slotInfo: "Wrap each example in the model's own chat format before training.",
                  }}
                  disabled={disabled}
                />
                <ControlledSelect
                  useControllerProps={{ name: 'unsloth.hardware.precision', control }}
                  formFieldProps={{
                    slotLabel: 'Precision',
                    slotInfo: 'Mixed-precision dtype for training. bf16 recommended for Ampere+.',
                  }}
                  items={UNSLOTH_PRECISION_ITEMS}
                  disabled={disabled}
                />
                <ControlledSelect
                  useControllerProps={{
                    name: 'unsloth.training.use_gradient_checkpointing',
                    control,
                  }}
                  formFieldProps={{
                    slotLabel: 'Gradient Checkpointing',
                    slotInfo:
                      'Recompute activations to save memory. Unsloth’s own kernel is the cheapest of the three.',
                  }}
                  items={UNSLOTH_GRADIENT_CHECKPOINTING_ITEMS}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.model.load_in_4bit', control }}
                  onChange={(checked) => {
                    if (checked) setValue('unsloth.model.load_in_8bit', false);
                  }}
                  formFieldProps={{
                    slotLabel: 'Load in 4-bit',
                    labelPosition: 'left',
                    slotInfo:
                      'Quantise the base model to 4-bit on load. On by default, and the reason unsloth fits large models on one GPU.',
                  }}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.model.load_in_8bit', control }}
                  onChange={(checked) => {
                    if (checked) setValue('unsloth.model.load_in_4bit', false);
                  }}
                  formFieldProps={{
                    slotLabel: 'Load in 8-bit',
                    labelPosition: 'left',
                    slotInfo: 'Quantise to 8-bit instead — more accurate than 4-bit, more VRAM.',
                  }}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.model.trust_remote_code', control }}
                  formFieldProps={{
                    slotLabel: 'Trust Remote Code',
                    labelPosition: 'left',
                    slotInfo:
                      'Allow the model repo to execute its own modelling code. Only enable for a source you trust.',
                  }}
                  disabled={disabled}
                />
                <ControlledTextInput
                  useControllerProps={{ name: 'unsloth.model.dtype', control }}
                  formFieldProps={{
                    slotLabel: 'Model dtype',
                    slotInfo: 'Leave unset for `auto`, which follows the hardware.',
                  }}
                  placeholder="auto"
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{ name: 'unsloth.model.rope_scaling', control }}
                  formFieldProps={{
                    slotLabel: 'RoPE Scaling (JSON)',
                    slotInfo:
                      'RoPE scaling config for long-context extension, passed to FastLanguageModel.from_pretrained (e.g. {"type": "linear", "factor": 2.0}). Leave blank to use the model\'s native context length.',
                  }}
                  placeholder='{ "type": "linear", "factor": 2.0 }'
                  disabled={disabled}
                />
              </Stack>
            </AccordionContent>
          </AccordionItem>
        </AccordionRoot>
      </Stack>
    </FormSection>
  );
};
