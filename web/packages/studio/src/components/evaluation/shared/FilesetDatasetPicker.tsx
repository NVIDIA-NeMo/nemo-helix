// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FormField, Select, Stack } from '@nvidia/foundations-react-core';
import { formatFromFileName } from '@studio/components/FileRowEditor/parse';
import { type ReactElement } from 'react';
import {
  type Control,
  type FieldValues,
  type Path,
  useController,
  useWatch,
} from 'react-hook-form';

const DATASET_FORMATS = ['json', 'jsonl', 'csv', 'parquet'];

const filesetOption = (fileset: { name: string }) => ({
  value: fileset.name,
  label: fileset.name,
});

interface FilesetDatasetPickerProps<T extends FieldValues> {
  workspace: string;
  control: Control<T>;
  /** Form field holding the chosen fileset name. */
  filesetName: Path<T>;
  /** Form field holding the chosen file's path within that fileset. */
  fileName: Path<T>;
  disabled?: boolean;
  /** The chosen file's content is still being read. */
  loading?: boolean;
  error?: string;
}

export function FilesetDatasetPicker<T extends FieldValues>({
  workspace,
  control,
  filesetName,
  fileName,
  disabled,
  loading,
  error,
}: FilesetDatasetPickerProps<T>): ReactElement {
  const fileset: string = useWatch({ control, name: filesetName });
  const { field: fileField } = useController({ control, name: fileName });

  const filesQuery = useFilesListFilesetFiles(workspace, fileset, undefined, {
    query: { enabled: !!fileset },
  });

  const fileItems = (filesQuery.data?.data ?? [])
    .filter((file) => DATASET_FORMATS.includes(formatFromFileName(file.path)))
    .map((file) => ({ value: file.path, children: file.path }));

  const noDatasetFiles = !!fileset && !filesQuery.isLoading && fileItems.length === 0;
  const placeholder = filesQuery.isLoading || loading ? 'Loading files...' : 'Select a file';

  return (
    <Stack gap="density-sm">
      <FilesetSearchableSelect<T>
        workspace={workspace}
        useControllerProps={{ control, name: filesetName }}
        formFieldProps={{ slotLabel: 'Fileset' }}
        renderOption={filesetOption}
        onChange={() => fileField.onChange('')}
        disabled={disabled}
      />
      <FormField
        slotLabel="File"
        slotHelp={
          noDatasetFiles
            ? 'This fileset has no JSONL, JSON, CSV, or Parquet files.'
            : 'JSONL, JSON, CSV, or Parquet. If the output is split across several files, only the file you pick is evaluated.'
        }
        slotError={error}
        status={error ? 'error' : undefined}
      >
        <Select
          disabled={disabled || !fileset}
          items={fileItems}
          value={fileField.value}
          onValueChange={fileField.onChange}
          placeholder={placeholder}
        />
      </FormField>
    </Stack>
  );
}
