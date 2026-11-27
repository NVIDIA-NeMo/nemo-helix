// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import {
  FilesetFilePicker,
  type FilesetFilePickerProps,
  type FilesetFileSelection,
  type FilesetFileValue,
} from '@nemo/common/src/components/FilesetPicker/FilesetFilePicker';
import { type ReactElement, useState } from 'react';
import { type FieldValues, useController, type UseControllerProps } from 'react-hook-form';

export interface ControlledFilesetFilePickerProps<T extends FieldValues> extends Omit<
  FilesetFilePickerProps,
  'value' | 'onChange' | 'onBlur' | 'errorText'
> {
  useControllerProps: UseControllerProps<T>;
  onChange?: (value: FilesetFileValue, selection: FilesetFileSelection) => void;
}

const toValue = (ref: unknown, workspace: string): FilesetFileValue | null => {
  if (typeof ref !== 'string' || !ref) return null;
  const parsed = parseFilesetLocation(ref, workspace);
  if (!parsed) return null;
  return { filesetRef: `${parsed.workspace}/${parsed.name}`, filePath: parsed.objectPath || null };
};

/**
 * Binds {@link FilesetFilePicker} to a field holding `workspace/name#path`. The field stays null
 * until a file is picked, so a fileset without a file never passes a non-empty check.
 */
export function ControlledFilesetFilePicker<T extends FieldValues>({
  useControllerProps,
  onChange,
  ...pickerProps
}: ControlledFilesetFilePickerProps<T>): ReactElement {
  const {
    field,
    fieldState: { error },
  } = useController(useControllerProps);
  // A fileset picked without a file isn't in the field yet. It is kept with the field value it was
  // written beside, so a reset or any other outside write drops it.
  const [pending, setPending] = useState<{ filesetRef: string | null; fieldValue: unknown }>({
    filesetRef: null,
    fieldValue: null,
  });

  const value = toValue(field.value, pickerProps.workspace) ?? {
    filesetRef: pending.fieldValue === field.value ? pending.filesetRef : null,
    filePath: null,
  };

  return (
    <FilesetFilePicker
      {...pickerProps}
      value={value}
      onChange={(next, selection) => {
        const fieldValue =
          next.filesetRef && next.filePath ? `${next.filesetRef}#${next.filePath}` : null;
        setPending({ filesetRef: next.filesetRef, fieldValue });
        field.onChange(fieldValue);
        onChange?.(next, selection);
      }}
      onBlur={field.onBlur}
      errorText={error?.message}
    />
  );
}
