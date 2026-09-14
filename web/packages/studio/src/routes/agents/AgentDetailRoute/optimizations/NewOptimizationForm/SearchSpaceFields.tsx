// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import type { OptimizationFormValues } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { type FC } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * Editable path and range for each parameter the intent chose to sweep.
 *
 * Behind the Advanced panel because the intent already answered this for most users; it is here
 * for the cases the defaults cannot know about — a different LLM block name in the agent's config,
 * or a range that has to stay inside a provider's limit.
 */
export const SearchSpaceFields: FC = () => {
  const { control, formState } = useFormContext<OptimizationFormValues>();
  const searchSpace = useWatch({ control, name: 'searchSpace' });
  const disabled = formState.isSubmitting;

  return (
    <Stack gap="density-lg" className="pt-density-md">
      {searchSpace.map((parameter, index) => (
        <Stack key={index} gap="density-sm">
          <ControlledTextInput
            useControllerProps={{ control, name: `searchSpace.${index}.path` }}
            label="Parameter"
            disabled={disabled}
          />
          <Flex gap="density-md" align="start">
            <ControlledTextInput
              useControllerProps={{ control, name: `searchSpace.${index}.low` }}
              label="Min"
              type="number"
              step={parameter.type === 'float' ? 0.1 : 1}
              disabled={disabled}
              formFieldProps={{ className: 'flex-1' }}
            />
            <ControlledTextInput
              useControllerProps={{ control, name: `searchSpace.${index}.high` }}
              label="Max"
              type="number"
              step={parameter.type === 'float' ? 0.1 : 1}
              disabled={disabled}
              formFieldProps={{ className: 'flex-1' }}
            />
          </Flex>
        </Stack>
      ))}
    </Stack>
  );
};
