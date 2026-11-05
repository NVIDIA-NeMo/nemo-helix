// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { FormField, Select } from '@nvidia/foundations-react-core';
import type { ComponentProps, ReactNode } from 'react';
import { useController } from 'react-hook-form';

// Generic type to handle both single and multiple selection
type SelectValue<T extends 'single' | 'multiple'> = T extends 'single' ? string : string[];
type SelectRenderValue<T extends 'single' | 'multiple'> = T extends 'single'
  ? (value: string) => ReactNode
  : (value: string[]) => ReactNode;

interface Props<T extends 'single' | 'multiple' = 'single'>
  // Omit ensures type safety and avoids uncontrolled component state
  extends
    Omit<ComponentProps<typeof Select>, 'onChange' | 'defaultValue'>,
    UseControllerComponentProps {
  hideError?: boolean;
  loading?: boolean;
  onChange?: (value: SelectValue<T>) => void;
  kind?: T;
  renderValue?: SelectRenderValue<T>;
}

export const ControlledSelect = <T extends 'single' | 'multiple' = 'single'>({
  kind = 'single' as T,
  loading,
  hideError,
  onChange,
  renderValue,
  formFieldProps,
  useControllerProps,
  ...selectProps
}: Props<T>) => {
  const {
    field: { onBlur, onChange: onChangeControl, value },
    fieldState: { error },
  } = useController(useControllerProps);

  const handleSelectedValueChangeSingle = (newValue: string) => {
    onChange?.(newValue as SelectValue<T>);
    onChangeControl(newValue);
  };

  const handleSelectedValueChangeMultiple = (newValue: string[]) => {
    onChange?.(newValue as SelectValue<T>);
    onChangeControl(newValue);
  };

  return (
    <FormField
      name={useControllerProps.name}
      slotError={hideError ? undefined : error?.message}
      status={error && 'error'}
      {...formFieldProps}
    >
      {() => {
        const baseProps = {
          value,
          onBlur,
          slotEnd: loading && <TextInputSpinner />,
          ...selectProps,
        };
        if (kind === 'multiple') {
          return (
            <Select
              {...baseProps}
              multiple
              onValueChange={handleSelectedValueChangeMultiple}
              renderValue={renderValue as SelectRenderValue<'multiple'>}
            />
          );
        } else {
          return (
            <Select
              {...baseProps}
              multiple={false}
              onValueChange={handleSelectedValueChangeSingle}
              renderValue={renderValue as SelectRenderValue<'single'>}
            />
          );
        }
      }}
    </FormField>
  );
};
