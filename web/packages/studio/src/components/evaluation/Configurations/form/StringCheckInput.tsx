// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Flex,
  FormField,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { EvaluationTargetMode } from '@studio/api/evaluation/types';
import type { CreateConfigFormData } from '@studio/hooks/evaluation/useCreateConfigurationForm';
import { FC } from 'react';
import { Controller, useFormContext, useWatch } from 'react-hook-form';

const STRING_CHECK_OPERATORS = ['equals', '!=', 'contains', 'startswith', 'endswith'] as const;

export interface StringCheckInputProps {
  disabled?: boolean;
  operatorLabel?: string;
}

export const StringCheckInput: FC<StringCheckInputProps> = ({
  disabled,
  operatorLabel = 'Operator',
}) => {
  const { control } = useFormContext<CreateConfigFormData>();

  // Watch target mode to determine label
  const targetMode = useWatch({ control, name: 'configData.targetMode' });
  const isOfflineMode = targetMode === EvaluationTargetMode.OFFLINE;
  const outputLabel = isOfflineMode ? 'Cached Output' : 'Actual Response';

  return (
    <Flex gap="density-md" align="center" justify="center" className="w-full">
      <Stack gap="density-xs">
        <Text kind="label/bold/md">{outputLabel}</Text>
      </Stack>

      <Stack className="w-[200px]">
        <Controller<CreateConfigFormData, 'configData.metricConfigs.stringCheck.operator'>
          name="configData.metricConfigs.stringCheck.operator"
          control={control}
          render={({ field, fieldState }) => (
            <FormField
              slotLabel={operatorLabel}
              slotError={fieldState.error?.message}
              status={fieldState.error ? 'error' : undefined}
            >
              <SelectRoot
                value={field.value || 'equals'}
                onValueChange={field.onChange}
                disabled={disabled}
              >
                <SelectTrigger placeholder="Select operator" />
                <SelectContent>
                  <SelectListbox>
                    {STRING_CHECK_OPERATORS.map((operator) => (
                      <SelectItem key={operator} value={operator}>
                        {operator}
                      </SelectItem>
                    ))}
                  </SelectListbox>
                </SelectContent>
              </SelectRoot>
            </FormField>
          )}
        />
      </Stack>

      <Stack gap="density-xs">
        <Text kind="label/bold/md">Ground Truth</Text>
      </Stack>
    </Flex>
  );
};
