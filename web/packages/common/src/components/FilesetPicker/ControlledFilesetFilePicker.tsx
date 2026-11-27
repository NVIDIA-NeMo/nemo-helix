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

interface PendingFileset {
  filesetRef: string;
  fieldValue: unknown;
  defaultValues: unknown;
}

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
    formState: { defaultValues },
  } = useController(useControllerProps);
  // A fileset picked without a file isn't in the field yet, so it is held here, tagged with the
  // field value it was written beside and the form's defaults at the time. Any other field value
  // (an outside write) or new defaults (`reset(values)`) drop it for good; adjusting state during
  // render keeps that free of an effect. A bare `reset()` to unchanged defaults leaves no trace in
  // public form state, so it can't be told apart from no change.
  const [pending, setPending] = useState<PendingFileset | null>(null);
  if (pending && (pending.fieldValue !== field.value || pending.defaultValues !== defaultValues)) {
    setPending(null);
  }

  const value = toValue(field.value, pickerProps.workspace) ?? {
    filesetRef: pending?.filesetRef ?? null,
    filePath: null,
  };

  return (
    <FilesetFilePicker
      {...pickerProps}
      value={value}
      onChange={(next, selection) => {
        const fieldValue =
          next.filesetRef && next.filePath ? `${next.filesetRef}#${next.filePath}` : null;
        setPending(
          next.filesetRef ? { filesetRef: next.filesetRef, fieldValue, defaultValues } : null
        );
        field.onChange(fieldValue);
        onChange?.(next, selection);
      }}
      onBlur={field.onBlur}
      errorText={error?.message}
    />
  );
}
