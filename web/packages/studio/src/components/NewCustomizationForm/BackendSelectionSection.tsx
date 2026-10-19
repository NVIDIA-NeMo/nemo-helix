// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { RadioGroupRoot, Text } from '@nvidia/foundations-react-core';
import { FormSection } from '@studio/components/NewCustomizationForm/FormSection';
import type { CustomizationFormFields } from '@studio/util/forms/customization';
import { useFormContext } from 'react-hook-form';

const BACKENDS = [
  {
    value: 'automodel' as const,
    title: 'Automodel',
    description:
      'Multi-GPU distributed training. Supports SFT and knowledge distillation with LoRA or full-weight fine-tuning.',
  },
  {
    value: 'unsloth' as const,
    title: 'Unsloth',
    description:
      'Single-GPU, memory-efficient training via 4-bit quantization. Ideal for smaller hardware with fast iteration.',
  },
  {
    value: 'rl' as const,
    title: 'RL',
    description:
      'Reinforcement learning via DPO on preference pairs or GRPO with a reward environment.',
  },
];

export const BackendSelectionSection = () => {
  const { watch, setValue, formState } = useFormContext<CustomizationFormFields>();
  const backend = watch('backend');
  const disabled = formState.isSubmitting;

  return (
    <FormSection title="Training Backend">
      <RadioGroupRoot
        name="backend"
        value={backend}
        onValueChange={(v) =>
          setValue('backend', v as CustomizationFormFields['backend'], { shouldValidate: false })
        }
        className="w-full"
        disabled={disabled}
      >
        <div className="grid grid-cols-2 gap-4">
          {BACKENDS.map((b) => (
            <RadioCard
              key={b.value}
              value={b.value}
              label={<Text kind="body/bold/lg">{b.title}</Text>}
              description={
                <Text className="text-secondary" kind="body/regular/md">
                  {b.description}
                </Text>
              }
              labelSide="left"
            />
          ))}
        </div>
      </RadioGroupRoot>
    </FormSection>
  );
};
