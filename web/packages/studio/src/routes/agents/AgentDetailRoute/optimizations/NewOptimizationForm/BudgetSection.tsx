// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RadioCard } from '@nemo/common/src/components/RadioCard';
import { Grid, RadioGroupRoot, Text } from '@nvidia/foundations-react-core';
import type { OptimizationFormValues } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import {
  type BudgetId,
  OPTIMIZATION_BUDGETS,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { type FC } from 'react';
import { useFormContext } from 'react-hook-form';

/** How many trials the sampler gets. More trials cost more agent runs but raise the odds the study
 *  finds a config that beats the current one. */
export const BudgetSection: FC = () => {
  const { watch, setValue, formState } = useFormContext<OptimizationFormValues>();
  const budget = watch('budget');

  return (
    <RadioGroupRoot
      name="budget"
      value={budget}
      disabled={formState.isSubmitting}
      className="w-full"
      onValueChange={(value) => setValue('budget', value as BudgetId, { shouldValidate: true })}
    >
      <Grid cols={{ base: 1, sm: 3 }} gap="4">
        {OPTIMIZATION_BUDGETS.map((option) => (
          <RadioCard
            key={option.id}
            value={option.id}
            checked={option.id === budget}
            labelSide="left"
            label={<Text kind="body/bold/md">{option.title}</Text>}
            description={
              <Text kind="body/regular/sm" color="secondary">
                {option.trials} trials · ~{option.estimatedMinutes} min
              </Text>
            }
          />
        ))}
      </Grid>
    </RadioGroupRoot>
  );
};
