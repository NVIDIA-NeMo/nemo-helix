// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import type { ReactElement } from 'react';
import { type FieldValues, useController, type UseControllerProps } from 'react-hook-form';

export interface ControlledFilesetPickerProps<T extends FieldValues> extends Omit<
  FilesetPickerProps,
  'value' | 'onChange' | 'onBlur' | 'errorText'
> {
  useControllerProps: UseControllerProps<T>;
  onChange?: (value: string | null, fileset?: FilesetOutput) => void;
}

export function ControlledFilesetPicker<T extends FieldValues>({
  useControllerProps,
  onChange,
  ...pickerProps
}: ControlledFilesetPickerProps<T>): ReactElement {
  const {
    field,
    fieldState: { error },
  } = useController(useControllerProps);

  return (
    <FilesetPicker
      {...pickerProps}
      value={(field.value as string | null | undefined) || null}
      onChange={(value, fileset) => {
        field.onChange(value);
        onChange?.(value, fileset);
      }}
      onBlur={field.onBlur}
      errorText={error?.message}
    />
  );
}
