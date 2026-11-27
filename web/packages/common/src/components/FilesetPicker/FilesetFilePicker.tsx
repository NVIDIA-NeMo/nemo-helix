// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetPicker, type FilesetPickerProps } from '@nemo/common/src/components/FilesetPicker';
import { FilesetFileCombobox } from '@nemo/common/src/components/FilesetPicker/FilesetFileCombobox';
import { DEFAULT_FILESET_PICKER_NOUN } from '@nemo/common/src/components/FilesetPicker/types';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { isAcceptedFileType } from '@nemo/common/src/utils/file';
import {
  filesListFilesetFiles,
  getFilesListFilesetFilesQueryKey,
} from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { FormField, Stack } from '@nvidia/foundations-react-core';
import { useQuery } from '@tanstack/react-query';
import { type FC, type ReactNode, useId, useMemo } from 'react';

export interface FilesetFileValue {
  /** `workspace/name`, or null. */
  filesetRef: string | null;
  /** Path inside the fileset, or null. Always null while `filesetRef` is null. */
  filePath: string | null;
}

export interface FilesetFileSelection {
  fileset?: FilesetOutput;
  file?: FilesetFileOutput;
}

export interface FilesetFilePickerProps extends Omit<
  FilesetPickerProps,
  'value' | 'onChange' | 'errorText'
> {
  value: FilesetFileValue;
  onChange: (value: FilesetFileValue, selection: FilesetFileSelection) => void;
  fileLabel?: ReactNode;
  /** Files with other extensions stay listed but can't be picked. Omit to accept every file. */
  acceptedFileExtensions?: readonly string[];
  /** Shown on the file field once a fileset is picked, otherwise on the fileset field. */
  errorText?: string;
}

export const FilesetFilePicker: FC<FilesetFilePickerProps> = ({
  value,
  onChange,
  fileLabel = 'File',
  acceptedFileExtensions,
  errorText,
  noun = DEFAULT_FILESET_PICKER_NOUN,
  ...filesetPickerProps
}) => {
  const fileFieldId = useId();
  const { workspace, name } = getPartsFromReference(value.filesetRef ?? '');
  const { data, isLoading, isError } = useQuery({
    queryKey: getFilesListFilesetFilesQueryKey(workspace, name),
    queryFn: ({ signal }) => filesListFilesetFiles(workspace, name, undefined, signal),
    enabled: Boolean(value.filesetRef && name),
  });
  const files = useMemo(
    () => [...(data?.data ?? [])].sort((a, b) => a.path.localeCompare(b.path)),
    [data]
  );

  // A saved path can outlive its file; say so instead of leaving the input blank.
  const fileMissing = Boolean(
    value.filePath && data && !files.some((file) => file.path === value.filePath)
  );
  const fileError =
    errorText ??
    (fileMissing ? `This file is no longer in this ${noun.one}. Select another one.` : undefined);

  const extensions = acceptedFileExtensions?.join(' or ');
  const eligibleCount = files.filter((file) =>
    isAcceptedFileType(file.path, acceptedFileExtensions)
  ).length;
  const fileHelp =
    isLoading || isError
      ? undefined
      : files.length === 0
        ? `This ${noun.one} has no files`
        : eligibleCount === 0
          ? `This ${noun.one} has no ${extensions} files`
          : extensions
            ? `Select a ${extensions} file`
            : undefined;

  return (
    <Stack gap="3">
      <FilesetPicker
        {...filesetPickerProps}
        // Once a fileset is picked the File field is what's left to fill, so only it reports blur;
        // leaving the fileset field then mustn't mark the value touched before a file is chosen.
        onBlur={value.filesetRef ? undefined : filesetPickerProps.onBlur}
        noun={noun}
        value={value.filesetRef}
        errorText={value.filesetRef ? undefined : errorText}
        onChange={(filesetRef, fileset) => onChange({ filesetRef, filePath: null }, { fileset })}
      />
      {value.filesetRef ? (
        <FormField
          id={fileFieldId}
          slotLabel={fileLabel}
          slotHelp={fileHelp}
          slotError={fileError}
          status={fileError ? 'error' : undefined}
          required={filesetPickerProps.required}
        >
          {({ status }) => (
            <FilesetFileCombobox
              id={fileFieldId}
              files={files}
              isLoading={isLoading}
              isError={isError}
              value={value.filePath}
              onChange={(filePath, file) =>
                onChange({ filesetRef: value.filesetRef, filePath }, { file })
              }
              acceptedFileExtensions={acceptedFileExtensions}
              noun={noun}
              disabled={filesetPickerProps.disabled}
              onBlur={filesetPickerProps.onBlur}
              status={status === 'error' ? 'error' : undefined}
            />
          )}
        </FormField>
      ) : null}
    </Stack>
  );
};
