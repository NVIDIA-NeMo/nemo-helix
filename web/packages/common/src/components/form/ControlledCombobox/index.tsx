// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { Combobox, type ComboboxProps, FormField } from '@nvidia/foundations-react-core';
import { type KeyboardEvent, type ReactNode, useState } from 'react';
import { useController } from 'react-hook-form';

// Generic type to handle both single and multiple selection
type ComboboxValue<T extends 'single' | 'multiple'> = T extends 'single' ? string : string[];

interface Props<T extends 'single' | 'multiple' = 'single'>
  // Omit ensures type safety and avoids uncontrolled component state
  extends
    Omit<
      ComboboxProps,
      | 'onChange'
      | 'defaultValue'
      | 'multiple'
      | 'kind'
      | 'value'
      | 'onValueChange'
      | 'renderValue'
      | 'ref'
    >,
    UseControllerComponentProps {
  label?: string;
  loading?: boolean;
  onChange?: (value: ComboboxValue<T>) => void;
  width?: string;
  hideError?: boolean;
  kind?: T;
  renderValue?: (
    value: string | string[] | undefined,
    setValue:
      | ((nextValue: string | string[]) => void)
      | ((nextValueFunc: (prevValue: string | string[]) => string | string[]) => void)
  ) => ReactNode;
  /**
   * Allow users to input a value that is not in the list. Also causes the value to not reset on blur.
   * Single-select writes the typed value straight through; multi-select appends it on Enter.
   * @default false
   */
  freeForm?: boolean;
}

export const ControlledCombobox = <T extends 'single' | 'multiple' = 'single'>({
  items = [],
  label,
  loading,
  onChange,
  width,
  hideError = false,
  useControllerProps,
  formFieldProps,
  kind = 'single' as T,
  renderValue,
  freeForm = false,
  inputValue,
  onInputValueChange,
  ...comboboxProps
}: Props<T>) => {
  const {
    field: { onBlur, onChange: onChangeControl, value },
    fieldState: { error },
  } = useController(useControllerProps);
  const [dispValue, setDispValue] = useState(value);

  const [typedValue, setTypedValue] = useState('');

  const isMultiple = kind === 'multiple';
  const addsOnEnter = freeForm && isMultiple;
  const hasCallerInputValue = inputValue !== undefined;

  const handleSelectedValueChange = (newValue: string | string[]) => {
    setDispValue(newValue);
    setTypedValue('');
    onChange?.(newValue as ComboboxValue<T>);
    onChangeControl(newValue);
  };

  const handleTempValueChange = (newValue: string) => {
    onInputValueChange?.(newValue);
    if (hasCallerInputValue) return;
    if (addsOnEnter) {
      setTypedValue(newValue);
    } else if (freeForm) {
      handleSelectedValueChange(newValue);
    } else {
      setDispValue(newValue);
    }
  };

  const addTypedValue = () => {
    const entry = (hasCallerInputValue ? inputValue : typedValue)?.trim();
    if (!entry) return false;
    const current = Array.isArray(value) ? (value as string[]) : [];
    if (!current.includes(entry)) {
      handleSelectedValueChange([...current, entry]);
    }
    setTypedValue('');
    return true;
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    comboboxProps.attributes?.ComboboxTrigger?.onKeyDown?.(event);
    if (!addsOnEnter || event.key !== 'Enter' || event.defaultPrevented) return;
    if (addTypedValue()) event.preventDefault();
  };

  const resolvedInputValue = hasCallerInputValue
    ? inputValue
    : addsOnEnter
      ? typedValue
      : isMultiple
        ? ''
        : dispValue || '';
  return (
    <FormField
      name={useControllerProps.name}
      slotLabel={label}
      slotError={hideError ? undefined : error?.message}
      status={error && 'error'}
      {...formFieldProps}
    >
      {({ status }) => {
        const baseProps = {
          status,
          style: { width },
          id: useControllerProps.name,
          name: useControllerProps.name,
          items,
          inputValue: resolvedInputValue,
          onInputValueChange: handleTempValueChange,
          onBlur,
          slotEnd: loading && <TextInputSpinner />,
          resetValueOnBlur: !freeForm,
          renderValue,
          ...comboboxProps,
          attributes: {
            ComboboxTrigger: {
              spellCheck: false,
              ...comboboxProps?.attributes?.ComboboxTrigger,
              onKeyDown: handleKeyDown,
            },
          },
        };

        if (kind === 'multiple') {
          return (
            <Combobox
              {...baseProps}
              multiple
              value={value as string[]}
              onValueChange={(newValue: string[]) => handleSelectedValueChange(newValue)}
            />
          );
        } else {
          return (
            <Combobox
              {...baseProps}
              value={value as string}
              onValueChange={(newValue: string) => handleSelectedValueChange(newValue)}
            />
          );
        }
      }}
    </FormField>
  );
};
