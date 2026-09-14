// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { Badge, Flex, RadioGroupRoot, Stack, Text } from '@nvidia/foundations-react-core';
import type { OptimizationFormValues } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import {
  type IntentId,
  intentById,
  OPTIMIZATION_INTENTS,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { type FC } from 'react';
import { useFormContext } from 'react-hook-form';

/**
 * What the study is tuning for. Picking one sets both the objective and the search space, so this
 * is the only required decision the user makes about the sweep itself — everything downstream is
 * an override of what this choice implies.
 */
export const IntentSection: FC = () => {
  const { watch, setValue, formState } = useFormContext<OptimizationFormValues>();
  const intent = watch('intent');

  return (
    <RadioGroupRoot
      name="intent"
      value={intent}
      disabled={formState.isSubmitting}
      className="w-full"
      onValueChange={(value) => {
        setValue('intent', value as IntentId, { shouldValidate: true });
        setValue('searchSpace', intentById(value as IntentId).parameters, { shouldValidate: true });
      }}
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {OPTIMIZATION_INTENTS.map((option) => (
          <RadioCard
            key={option.id}
            value={option.id}
            checked={option.id === intent}
            labelSide="left"
            label={<Text kind="body/bold/md">{option.title}</Text>}
            description={
              <Stack gap="density-sm">
                <Text kind="body/regular/sm" color="secondary">
                  {option.description}
                </Text>
                <Flex align="center" gap="density-sm" wrap="wrap">
                  <Text kind="body/regular/xs" color="secondary">
                    Sweeps
                  </Text>
                  {option.parameters.map((parameter) => (
                    <Badge key={parameter.path} color="gray" kind="outline">
                      {parameter.label}
                    </Badge>
                  ))}
                </Flex>
              </Stack>
            }
          />
        ))}
      </div>
    </RadioGroupRoot>
  );
};
