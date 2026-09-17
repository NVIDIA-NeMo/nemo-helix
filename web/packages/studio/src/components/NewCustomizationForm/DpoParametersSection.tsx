// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledSliderWithTextInput } from '@nemo/common/src/components/form/ControlledSliderWithTextInput';
import { ControlledSwitch } from '@nemo/common/src/components/form/ControlledSwitch';
import {
  AccordionContent,
  AccordionItem,
  AccordionRoot,
  AccordionTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { DPO_SPEC_DEFAULTS, specSliderProps } from '@studio/util/forms/specDefaults';
import { useFormContext } from 'react-hook-form';

export const DpoParametersSection = () => {
  const { control, formState } = useFormContext<CustomizationFormFields>();
  const disabled = formState.isSubmitting;

  return (
    <FormSection title="DPO Parameters">
      <Stack gap="density-lg">
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'rl.training.ref_policy_kl_penalty', control }}
          formFieldProps={{
            slotLabel: 'KL Penalty (β)',
            slotInfo:
              'KL divergence coefficient from the DPO paper. Higher values keep the fine-tuned model closer to the reference policy.',
          }}
          {...specSliderProps(DPO_SPEC_DEFAULTS, 'ref_policy_kl_penalty')}
          min={0}
          max={1}
          step={0.01}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'rl.training.preference_loss_weight', control }}
          formFieldProps={{
            slotLabel: 'Preference Loss Weight',
            slotInfo: 'Scaling factor for the DPO preference (chosen vs rejected) loss term.',
          }}
          {...specSliderProps(DPO_SPEC_DEFAULTS, 'preference_loss_weight')}
          min={0}
          max={10}
          step={0.1}
          disabled={disabled}
        />
        <ControlledSliderWithTextInput
          useControllerProps={{ name: 'rl.training.sft_loss_weight', control }}
          formFieldProps={{
            slotLabel: 'SFT Regularization Loss Weight',
            slotInfo:
              'Weight for the SFT (imitation) regularization loss on the chosen response. Set to 0 to disable.',
          }}
          {...specSliderProps(DPO_SPEC_DEFAULTS, 'sft_loss_weight')}
          min={0}
          max={10}
          step={0.1}
          disabled={disabled}
        />
        <AccordionRoot multiple>
          <AccordionItem value="advanced-dpo" className="border-b-0">
            <AccordionTrigger>
              <Text kind="label/bold/md">Advanced</Text>
            </AccordionTrigger>
            <AccordionContent>
              <Stack gap="density-md" className="pt-density-md">
                <ControlledSliderWithTextInput
                  useControllerProps={{ name: 'rl.training.max_grad_norm', control }}
                  formFieldProps={{
                    slotLabel: 'Max Gradient Norm',
                    slotInfo: 'Gradient clipping threshold.',
                  }}
                  {...specSliderProps(DPO_SPEC_DEFAULTS, 'max_grad_norm')}
                  min={0}
                  max={10}
                  step={0.1}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{
                    name: 'rl.training.preference_average_log_probs',
                    control,
                  }}
                  formFieldProps={{
                    slotLabel: 'Average Log-Probs (Preference)',
                    labelPosition: 'left',
                    slotInfo:
                      'Average log-probabilities across tokens when computing the preference loss instead of summing.',
                  }}
                  disabled={disabled}
                />
                <ControlledSwitch
                  useControllerProps={{ name: 'rl.training.sft_average_log_probs', control }}
                  formFieldProps={{
                    slotLabel: 'Average Log-Probs (SFT)',
                    labelPosition: 'left',
                    slotInfo:
                      'Average log-probabilities across tokens when computing the SFT regularization loss.',
                  }}
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
