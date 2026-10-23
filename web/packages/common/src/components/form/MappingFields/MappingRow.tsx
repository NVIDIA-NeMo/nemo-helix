/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { ControlledCombobox } from '@nemo/common/src/components/form/ControlledCombobox';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import type {
  KeyValueComboboxPassthrough,
  KeyValueTextInputPassthrough,
} from '@nemo/common/src/components/form/MappingFields/types';
import { Trash } from '@nemo/icons/icons';
import { Button, Flex } from '@nvidia/foundations-react-core';
import cn from 'classnames';
import { memo } from 'react';
import { Control, FieldValues } from 'react-hook-form';

interface Props<TFieldValues extends FieldValues> {
  control: Control<TFieldValues>;
  /** Field array path, e.g. `mappings`. */
  name: string;
  index: number;
  /** The trailing draft row cannot be removed. */
  isLastRow: boolean;
  isDisabled: boolean;
  keyOpts: string[];
  valueOpts: string[];
  keyColumnLabel: string;
  valueColumnLabel: string;
  keyCombobox: Partial<KeyValueComboboxPassthrough>;
  valueCombobox: Partial<KeyValueComboboxPassthrough>;
  keyTextInput: Partial<KeyValueTextInputPassthrough>;
  valueTextInput: Partial<KeyValueTextInputPassthrough>;
  onRemove: (index: number) => void;
}

const MappingRowInner = <TFieldValues extends FieldValues>({
  control,
  name,
  index,
  isLastRow,
  isDisabled,
  keyOpts,
  valueOpts,
  keyColumnLabel,
  valueColumnLabel,
  keyCombobox,
  valueCombobox,
  keyTextInput,
  valueTextInput,
  onRemove,
}: Props<TFieldValues>) => {
  const {
    formFieldProps: keyComboboxFormFieldProps,
    className: keyComboboxClassName,
    attributes: keyComboboxAttributes,
    ...keyComboboxRest
  } = keyCombobox;

  const {
    formFieldProps: valueComboboxFormFieldProps,
    className: valueComboboxClassName,
    attributes: valueComboboxAttributes,
    ...valueComboboxRest
  } = valueCombobox;

  const {
    formFieldProps: keyTextFormFieldProps,
    className: keyTextClassName,
    hideError: keyTextHideError,
    attributes: keyTextAttributes,
    ...keyTextRest
  } = keyTextInput;

  const {
    formFieldProps: valueTextFormFieldProps,
    className: valueTextClassName,
    hideError: valueTextHideError,
    attributes: valueTextAttributes,
    ...valueTextRest
  } = valueTextInput;

  return (
    <Flex gap="density-lg" align="end" justify="between">
      {keyOpts.length > 0 ? (
        <ControlledCombobox
          {...keyComboboxRest}
          disabled={isDisabled}
          freeForm
          dismissible={false}
          hideError
          className={cn('font-normal', keyComboboxClassName)}
          attributes={keyComboboxAttributes}
          formFieldProps={{
            className: 'min-w-0 flex-1 font-bold',
            ...keyComboboxFormFieldProps,
          }}
          useControllerProps={{ control, name: `${name}.${index}.key`, disabled: isDisabled }}
          items={keyOpts}
          label={index === 0 ? keyColumnLabel : ''}
        />
      ) : (
        <ControlledTextInput
          {...keyTextRest}
          disabled={isDisabled}
          hideError={keyTextHideError ?? true}
          className={keyTextClassName}
          attributes={keyTextAttributes}
          formFieldProps={{
            className: 'min-w-0 flex-1',
            ...keyTextFormFieldProps,
          }}
          useControllerProps={{ control, name: `${name}.${index}.key`, disabled: isDisabled }}
          label={index === 0 ? keyColumnLabel : ''}
        />
      )}
      {valueOpts.length > 0 ? (
        <ControlledCombobox
          {...valueComboboxRest}
          disabled={isDisabled}
          freeForm
          dismissible={false}
          hideError
          className={cn('font-normal', valueComboboxClassName)}
          attributes={valueComboboxAttributes}
          formFieldProps={{
            className: 'min-w-0 flex-1 font-bold',
            ...valueComboboxFormFieldProps,
          }}
          useControllerProps={{ control, name: `${name}.${index}.value`, disabled: isDisabled }}
          items={valueOpts}
          label={index === 0 ? valueColumnLabel : ''}
        />
      ) : (
        <ControlledTextInput
          {...valueTextRest}
          disabled={isDisabled}
          hideError={valueTextHideError ?? true}
          className={valueTextClassName}
          attributes={valueTextAttributes}
          formFieldProps={{
            className: 'min-w-0 flex-1',
            ...valueTextFormFieldProps,
          }}
          useControllerProps={{ control, name: `${name}.${index}.value`, disabled: isDisabled }}
          label={index === 0 ? valueColumnLabel : ''}
        />
      )}
      <Button
        type="button"
        kind="tertiary"
        aria-label="Remove row"
        disabled={isDisabled || isLastRow}
        onClick={() => {
          onRemove(index);
        }}
      >
        <Trash />
      </Button>
    </Flex>
  );
};

/**
 * Memoized so that typing in one row (which re-renders the field array via `useWatch`)
 * does not re-render every other row's combobox.
 */
export const MappingRow = memo(MappingRowInner) as typeof MappingRowInner;
