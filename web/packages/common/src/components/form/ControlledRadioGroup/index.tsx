// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { FormField, RadioGroup } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC } from 'react';
import { useController } from 'react-hook-form';

interface Props
  extends Omit<ComponentProps<typeof RadioGroup>, 'name'>, UseControllerComponentProps {}

export const ControlledRadioGroup: FC<Props> = ({
  useControllerProps,
  formFieldProps,
  required,
  items,
  ...radioGroupProps
}) => {
  const { field, fieldState, formState } = useController(useControllerProps);
  const { error } = fieldState;
  const { isValid } = formState;
  return (
    <FormField
      slotError={error?.message}
      name={useControllerProps.name}
      required={required}
      status={error ? 'error' : isValid ? 'success' : undefined}
      {...formFieldProps}
    >
      <RadioGroup
        {...radioGroupProps}
        items={items}
        value={field.value}
        onValueChange={(value) => {
          field.onChange(value);
        }}
        name={useControllerProps.name}
      />
    </FormField>
  );
};
