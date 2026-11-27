// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetFileRow } from '@nemo/common/src/components/FilesetPicker/FilesetFileRow';
import type { FilesetPickerNoun } from '@nemo/common/src/components/FilesetPicker/types';
import { TextInputSpinner } from '@nemo/common/src/components/form/TextInputSpinner';
import { isAcceptedFileType } from '@nemo/common/src/utils/file';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { Combobox, type ComboboxEntry, Text } from '@nvidia/foundations-react-core';
import { type FC, useMemo } from 'react';

export interface FilesetFileComboboxProps {
  /** Sorted file listing of the selected fileset. */
  files: readonly FilesetFileOutput[];
  isLoading: boolean;
  isError: boolean;
  value: string | null;
  onChange: (path: string | null, file?: FilesetFileOutput) => void;
  acceptedFileExtensions?: readonly string[];
  noun: FilesetPickerNoun;
  disabled?: boolean;
  status?: 'error' | 'success';
  onBlur?: () => void;
  id?: string;
}

export const FilesetFileCombobox: FC<FilesetFileComboboxProps> = ({
  files,
  isLoading,
  isError,
  value,
  onChange,
  acceptedFileExtensions,
  noun,
  disabled,
  status,
  onBlur,
  id,
}) => {
  const ineligibleReason = acceptedFileExtensions?.length
    ? `Needs ${acceptedFileExtensions.join(' or ')}`
    : undefined;

  const items = useMemo<ComboboxEntry[]>(() => {
    const entries: ComboboxEntry[] = files.map((file) => {
      const eligible = isAcceptedFileType(file.path, acceptedFileExtensions);
      return {
        value: file.path,
        filterValue: file.path,
        disabled: !eligible,
        children: (
          <FilesetFileRow file={file} ineligibleReason={eligible ? undefined : ineligibleReason} />
        ),
      };
    });
    // KUI only labels the input for a value that is an item, so a path missing from the listing
    // stays as a disabled row and the input keeps showing it.
    if (value && !isLoading && !files.some((file) => file.path === value)) {
      entries.unshift({
        value,
        filterValue: value,
        disabled: true,
        children: (
          <Text kind="body/regular/sm" className="text-secondary">
            {value}
          </Text>
        ),
      });
    }
    return entries;
  }, [files, acceptedFileExtensions, ineligibleReason, value, isLoading]);

  const emptyMessage = isError
    ? "Couldn't load files"
    : isLoading
      ? 'Loading files…'
      : files.length === 0
        ? `This ${noun.one} has no files`
        : 'No files match';

  return (
    <Combobox
      id={id}
      items={items}
      value={value ?? ''}
      onValueChange={(next: string) =>
        onChange(
          next || null,
          files.find((file) => file.path === next)
        )
      }
      emptyStateMessage={emptyMessage}
      placeholder="Select a file…"
      dismissible
      disabled={disabled}
      status={status}
      attributes={{ ComboboxTrigger: { onBlur } }}
      slotEnd={isLoading ? <TextInputSpinner /> : undefined}
    />
  );
};
