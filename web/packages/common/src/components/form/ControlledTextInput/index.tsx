// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { stringToNumber } from '@nemo/common/src/components/form/ControlledTextInput/utils';
import { MaskedTextInput } from '@nemo/common/src/components/form/MaskedTextInput';
import type { UseControllerComponentProps } from '@nemo/common/src/types';
import { Flex, FormField, TextInput } from '@nvidia/foundations-react-core';
import { type ComponentProps, type FocusEvent, type ReactNode, useCallback } from 'react';
import { useController } from 'react-hook-form';

interface Props extends ComponentProps<typeof TextInput>, UseControllerComponentProps {
  label?: ReactNode | string;
  slotEnd?: ReactNode;
  /**
   * When true, selects all text in the input when focused.
   * Useful for pre-populated inputs where users typically want to replace the entire value.
   */
  selectOnFocus?: boolean;
  /** When true, validation errors are not shown on the field (e.g. parent shows a summary). */
  hideError?: boolean;
  /**
   * When true, renders the value as a masked password field with a visibility toggle.
   * Incompatible with `type="number"`.
   */
  masked?: boolean;
}

export const ControlledTextInput = ({
  useControllerProps,
  label,
  'aria-label': ariaLabel,
  formFieldProps,
  required,
  slotEnd,
  selectOnFocus,
  hideError = false,
  masked = false,
  attributes: textInputAttributes,
  ...props
}: Props) => {
  const {
    field: { onBlur, onChange, value, disabled },
    fieldState: { error },
  } = useController(useControllerProps);
  const showError = !hideError && !!error;

  const wrappedOnChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const valueAsString = e.currentTarget.value;

    if (props.type === 'number') {
      const numberValue = stringToNumber(valueAsString);
      // Only update with the number if it's valid, otherwise keep the string for validation
      if (numberValue !== undefined) {
        onChange(numberValue);
      } else {
        // For invalid numbers, pass the string so validation can catch it
        onChange(valueAsString);
      }
    } else {
      onChange(valueAsString);
    }

    props?.onChange?.(e);
  };

  const wrappedOnFocus = useCallback(
    (e: FocusEvent<HTMLInputElement>) => {
      if (selectOnFocus) {
        e.target.select();
      }
      props?.onFocus?.(e);
    },
    [selectOnFocus, props]
  );

  return (
    <FormField
      name={useControllerProps.name}
      slotLabel={label}
      slotError={showError ? error?.message?.toString() : undefined}
      aria-label={
        typeof ariaLabel === 'string' ? ariaLabel : typeof label === 'string' ? label : undefined
      }
      status={showError ? 'error' : undefined}
      required={required}
      {...formFieldProps}
    >
      <Flex gap="density-sm" align="center" className="w-full">
        {masked ? (
          <MaskedTextInput
            attributes={textInputAttributes}
            disabled={disabled}
            value={value}
            status={showError ? 'error' : undefined}
            onBlur={onBlur}
            {...props}
            onChange={wrappedOnChange}
            onFocus={wrappedOnFocus}
          />
        ) : (
          <TextInput
            attributes={textInputAttributes}
            disabled={disabled}
            value={value}
            status={showError ? 'error' : undefined}
            onBlur={onBlur}
            {...props}
            onChange={wrappedOnChange}
            onFocus={wrappedOnFocus}
          />
        )}
        {slotEnd}
      </Flex>
    </FormField>
  );
};
