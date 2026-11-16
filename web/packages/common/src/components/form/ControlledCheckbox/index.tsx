// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { Checkbox, FormField } from '@nvidia/foundations-react-core';
import type { ComponentProps, FC } from 'react';
import { useController } from 'react-hook-form';

interface Props extends ComponentProps<typeof Checkbox>, UseControllerComponentProps {}

export const ControlledCheckbox: FC<Props> = ({
  useControllerProps,
  formFieldProps,
  ...checkboxProps
}) => {
  const { field, fieldState, formState } = useController(useControllerProps);
  const { error } = fieldState;
  const { isValid } = formState;
  return (
    <FormField
      slotError={error?.message}
      name={useControllerProps.name}
      status={error ? 'error' : isValid ? 'success' : undefined}
      {...formFieldProps}
    >
      <Checkbox
        checked={field.value ?? false}
        onCheckedChange={(checked) => {
          field.onChange(checked ?? false);
        }}
        attributes={{
          CheckboxInput: {
            className: 'cursor-pointer',
          },
        }}
        {...checkboxProps}
      />
    </FormField>
  );
};
