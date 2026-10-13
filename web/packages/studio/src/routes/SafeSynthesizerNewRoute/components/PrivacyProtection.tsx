// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  RadioGroupRoot,
  RadioGroupItem,
  RadioGroupInput,
  Text,
  Flex,
} from '@nvidia/foundations-react-core';
import { Controller, useFormContext } from 'react-hook-form';

export const PrivacyProtection = () => {
  const { control } = useFormContext();

  return (
    <Controller
      name="spec.config.privacy.dp_enabled"
      control={control}
      render={({ field }) => (
        <RadioGroupRoot
          name="privacy-level"
          value={field.value ? 'highest' : 'standard'}
          onValueChange={(value) => field.onChange(value === 'highest')}
          className="w-full"
        >
          {[
            {
              value: 'standard',
              label: 'Standard Privacy (Default)',
              description:
                'Uses the standard privacy inherent in synthetic data generation, balancing privacy and quality. This option is faster and generally more reliable.',
            },
            {
              value: 'highest',
              label: 'Highest Privacy (Advanced)',
              description:
                'Applies Differential Privacy, the gold standard of privacy, during training. This process adds noise to provide mathematical guarantees of privacy, but could result in lower quality results and/or longer training time.',
            },
          ].map((item) => (
            <RadioGroupItem
              className="w-full cursor-pointer items-start justify-start gap-2 rounded-md border-1 border-interaction-base bg-surface-raised px-3 py-4 transition-all data-[state=checked]:translate-y-[-2px] data-[state=checked]:border-transparent data-[state=checked]:font-bold data-[state=checked]:shadow-md data-[state=checked]:ring-2 data-[state=checked]:ring-brand [&_*]:cursor-pointer"
              key={item.value}
              aria-labelledby={`${item.value}-label`}
            >
              <RadioGroupInput value={item.value} id={item.value} />
              <Flex direction="col" align="start" gap="2" id={`${item.value}-label`}>
                <Flex align="center" gap="1">
                  <Text kind="label/regular/md">{item.label}</Text>
                </Flex>
                <Text kind="body/regular/sm" className="text-left text-secondary">
                  {item.description}
                </Text>
              </Flex>
            </RadioGroupItem>
          ))}
        </RadioGroupRoot>
      )}
    />
  );
};
