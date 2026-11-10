// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { FormField, TextArea } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';
import { useController } from 'react-hook-form';

interface Props
  extends Omit<ComponentProps<typeof TextArea>, 'onChange'>, UseControllerComponentProps {
  label?: string;
  onChange?: (value: string) => void;
}

export const ControlledTextArea = ({
  useControllerProps,
  label,
  formFieldProps,
  attributes,
  onChange,
  ...props
}: Props) => {
  const {
    field: { onChange: onChangeField, value, disabled },
    fieldState: { error },
  } = useController(useControllerProps);
  const handleValueChange = (next: string) => {
    onChange?.(next);
    onChangeField(next);
  };
  return (
    <FormField
      slotLabel={label}
      slotError={error?.message || ''}
      aria-label={label}
      status={error ? 'error' : undefined}
      {...formFieldProps}
    >
      <TextArea
        attributes={attributes}
        {...props}
        value={value}
        onValueChange={handleValueChange}
        disabled={props.disabled || disabled}
        status={error ? 'error' : undefined}
      />
    </FormField>
  );
};
