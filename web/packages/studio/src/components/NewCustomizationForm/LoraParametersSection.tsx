// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { ControlledJsonInput } from '@studio/components/NewCustomizationForm/ControlledJsonInput';
import { ControlledStringListInput } from '@studio/components/NewCustomizationForm/ControlledStringListInput';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import {
  AUTOMODEL_SPEC_DEFAULTS,
  UNSLOTH_SPEC_DEFAULTS,
  specSliderProps,
} from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

const INIT_LORA_WEIGHTS_OPTIONS = [
  { label: 'Default (true)', value: 'true' },
  { label: 'Off (false)', value: 'false' },
  { label: 'Gaussian', value: 'gaussian' },
  { label: 'PiSSA', value: 'pissa' },
  { label: 'OLoRA', value: 'olora' },
  { label: 'LoftQ', value: 'loftq' },
];

const RANK_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256];

const BIAS_OPTIONS = [
  { label: 'None', value: 'none' },
  { label: 'All', value: 'all' },
  { label: 'LoRA Only', value: 'lora_only' },
];

export const LoraParametersSection = () => {
  const { control, watch, setValue, formState } = useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  if (backend === 'automodel') {
    const rank = watch('automodel.training.lora.rank') ?? 16;
    return (
      <FormSection title="LoRA Parameters">
        <Stack gap="density-lg">
          <FormField slotLabel="Rank">
            {() => (
              <SelectRoot
                value={String(rank)}
                onValueChange={(v: string) =>
                  setValue('automodel.training.lora.rank', Number(v), { shouldValidate: true })
                }
                disabled={disabled}
              >
                <SelectTrigger placeholder="Select rank" aria-label="Rank" />
                <SelectContent>
                  <SelectListbox>
                    {RANK_VALUES.map((v) => (
                      <SelectItem key={v} value={String(v)}>
                        {v}
                      </SelectItem>
                    ))}
                  </SelectListbox>
                </SelectContent>
              </SelectRoot>
            )}
          </FormField>
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.training.lora.alpha', control }}
            formFieldProps={{ slotLabel: 'Alpha' }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'training_lora_alpha')}
            min={1}
            max={512}
            step={1}
            disabled={disabled}
          />
          <ControlledSliderWithTextInput
            useControllerProps={{ name: 'automodel.training.lora.dropout', control }}
            formFieldProps={{ slotLabel: 'Dropout' }}
            {...specSliderProps(AUTOMODEL_SPEC_DEFAULTS, 'training_lora_dropout')}
            min={0}
            max={1}
            step={0.01}
            disabled={disabled}
          />
          <ControlledSwitch
            useControllerProps={{ name: 'automodel.training.lora.merge', control }}
            formFieldProps={{ slotLabel: 'Merge weights after training', labelPosition: 'left' }}
            disabled={disabled}
          />
          <AccordionRoot multiple>
            <AccordionItem value="advanced-lora" className="border-b-0">
              <AccordionTrigger>
                <Text kind="label/bold/md">Advanced</Text>
              </AccordionTrigger>
              <AccordionContent>
                <Stack gap="density-md" className="pt-density-md">
                  <ControlledSwitch
                    useControllerProps={{ name: 'automodel.training.lora.use_triton', control }}
                    formFieldProps={{ slotLabel: 'Use Triton Kernels', labelPosition: 'left' }}
                    disabled={disabled}
                  />
                  <ControlledStringListInput
                    useControllerProps={{
                      name: 'automodel.training.lora.target_modules',
                      control,
                    }}
                    formFieldProps={{
                      slotLabel: 'Target Modules',
                      slotInfo:
                        'Modules to attach adapters to. Left unset, the backend picks them from the model architecture.',
                    }}
                    placeholder="q_proj, v_proj"
                    disabled={disabled}
                  />
                  <ControlledStringListInput
                    useControllerProps={{
                      name: 'automodel.training.lora.exclude_modules',
                      control,
                    }}
                    formFieldProps={{ slotLabel: 'Exclude Modules' }}
                    placeholder="*.out_proj"
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

  const rank = watch('unsloth.training.lora.rank') ?? 16;
  const bias = watch('unsloth.training.lora.bias') ?? 'none';
  const initLoraWeights = watch('unsloth.training.lora.init_lora_weights') ?? true;

  return (
    <FormSection title="LoRA Parameters">
      <Stack gap="density-lg">
        <FormField slotLabel="Rank">
          {() => (
            <SelectRoot
              value={String(rank)}
              onValueChange={(v: string) =>
                setValue('unsloth.training.lora.rank', Number(v), { shouldValidate: true })
              }
              disabled={disabled}
            >
              <SelectTrigger placeholder="Select rank" aria-label="Rank" />
              <SelectContent>
                <SelectListbox>
                  {RANK_VALUES.map((v) => (
                    <SelectItem key={v} value={String(v)}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectListbox>
              </SelectContent>
            </SelectRoot>
          )}
        </FormField>
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.training.lora.alpha', control }}
          formFieldProps={{ slotLabel: 'Alpha' }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'training_lora_alpha')}
          min={1}
          max={512}
          step={1}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'unsloth.training.lora.dropout', control }}
          formFieldProps={{ slotLabel: 'Dropout' }}
          {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'training_lora_dropout')}
          min={0}
          max={1}
          step={0.01}
          disabled={disabled}
        />
        <AccordionRoot multiple>
          <AccordionItem value="advanced-lora" className="border-b-0">
            <AccordionTrigger>
              <Text kind="label/bold/md">Advanced</Text>
            </AccordionTrigger>
            <AccordionContent>
              <Stack gap="density-md" className="pt-density-md">
                <FormField slotLabel="Bias">
                  {() => (
                    <SelectRoot
                      value={bias}
                      onValueChange={(v: string) =>
                        setValue('unsloth.training.lora.bias', v as 'none' | 'all' | 'lora_only', {
                          shouldValidate: true,
                        })
                      }
                      disabled={disabled}
                    >
                      <SelectTrigger placeholder="Select bias" aria-label="Bias" />
                      <SelectContent>
                        <SelectListbox>
                          {BIAS_OPTIONS.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectListbox>
                      </SelectContent>
                    </SelectRoot>
                  )}
                </FormField>
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.training.lora.use_rslora', control }}
                  formFieldProps={{
                    slotLabel: 'Use rsLoRA (rank-stabilized)',
                    labelPosition: 'left',
                  }}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'unsloth.training.lora.use_dora', control }}
                  formFieldProps={{ slotLabel: 'Use DoRA', labelPosition: 'left' }}
                  disabled={disabled}
                />
                <FormField slotLabel="Init LoRA Weights">
                  {() => (
                    <SelectRoot
                      value={String(initLoraWeights)}
                      onValueChange={(v: string) =>
                        setValue(
                          'unsloth.training.lora.init_lora_weights',
                          v === 'true' ? true : v === 'false' ? false : (v as 'gaussian'),
                          { shouldValidate: false }
                        )
                      }
                      disabled={disabled}
                    >
                      <SelectTrigger
                        placeholder="Select init scheme"
                        aria-label="Init LoRA Weights"
                      />
                      <SelectContent>
                        <SelectListbox>
                          {INIT_LORA_WEIGHTS_OPTIONS.map((opt) => (
                            <SelectItem key={opt.value} value={opt.value}>
                              {opt.label}
                            </SelectItem>
                          ))}
                        </SelectListbox>
                      </SelectContent>
                    </SelectRoot>
                  )}
                </FormField>
                <ControlledStringListInput
                  useControllerProps={{ name: 'unsloth.training.lora.target_modules', control }}
                  formFieldProps={{
                    slotLabel: 'Target Modules',
                    slotInfo:
                      'Modules to attach adapters to. Left unset, the backend picks them from the model architecture.',
                  }}
                  placeholder="q_proj, v_proj"
                  disabled={disabled}
                />
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'unsloth.training.lora.random_state', control }}
                  formFieldProps={{
                    slotLabel: 'Random State',
                    slotInfo: 'Seed for adapter initialisation.',
                  }}
                  {...specSliderProps(UNSLOTH_SPEC_DEFAULTS, 'training_lora_random_state')}
                  min={0}
                  max={999999}
                  step={1}
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{ name: 'unsloth.training.lora.modules_to_save', control }}
                  formFieldProps={{ slotLabel: 'Modules to Save (JSON array)' }}
                  placeholder='["lm_head", "embed_tokens"]'
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{ name: 'unsloth.training.lora.loftq_config', control }}
                  formFieldProps={{ slotLabel: 'LoftQ Config (JSON)' }}
                  placeholder='{ "loftq_bits": 4 }'
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{
                    name: 'unsloth.training.lora.layers_to_transform',
                    control,
                  }}
                  formFieldProps={{ slotLabel: 'Layers to Transform (JSON)' }}
                  placeholder="[0, 1, 2] or 5"
                  disabled={disabled}
                />
                <ControlledJsonInput
                  useControllerProps={{ name: 'unsloth.training.lora.layer_replication', control }}
                  formFieldProps={{ slotLabel: 'Layer Replication (JSON)' }}
                  placeholder="[[0, 8], [4, 12]]"
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
