// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FormField, Select, Stack } from '@nvidia/foundations-react-core';
import {
  JOB_LOGS_PREFIX,
  parquetBatchGroups,
} from '@studio/components/evaluation/shared/parquetBatchGroups';
import { formatFromFileName } from '@studio/components/FileRowEditor/parse';
import { MiddleTruncatedPath } from '@studio/components/MiddleTruncatedPath';
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
  /** Form field holding a glob over a folder of Parquet batches when that option is chosen, or ''
   *  for a single file. ``fileName`` then holds the folder's first batch, which is validated. */
  batchGlobName: Path<T>;
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
  batchGlobName,
  disabled,
  loading,
  error,
}: FilesetDatasetPickerProps<T>): ReactElement {
  const fileset: string = useWatch({ control, name: filesetName });
  const { field: fileField } = useController({ control, name: fileName });
  const { field: batchGlobField } = useController({ control, name: batchGlobName });

  const pickFile = (path: string) => {
    fileField.onChange(path);
    batchGlobField.onChange('');
  };

  const filesQuery = useFilesListFilesetFiles(workspace, fileset, undefined, {
    query: { enabled: !!fileset },
  });

  const filesetPaths = (filesQuery.data?.data ?? []).map((file) => file.path);
  const batchGroups = parquetBatchGroups(filesetPaths);

  const fileOptions = [
    ...batchGroups.map((group) => ({
      value: group.glob,
      label: `All ${group.count} Parquet files in ${group.dir || 'the fileset root'}`,
    })),
    ...filesetPaths
      .filter(
        (path) =>
          !path.startsWith(JOB_LOGS_PREFIX) && DATASET_FORMATS.includes(formatFromFileName(path))
      )
      .map((path) => ({ value: path, label: path })),
  ];
  const fileItems = fileOptions.map(({ value, label }) => ({
    value,
    filterValue: label,
    children: <MiddleTruncatedPath>{label}</MiddleTruncatedPath>,
  }));
  const renderFileValue = (value: string | string[] | undefined) => {
    const label = fileOptions.find((option) => option.value === value)?.label;
    return label ? <MiddleTruncatedPath>{label}</MiddleTruncatedPath> : undefined;
  };

  const pickItem = (value: string) => {
    const group = batchGroups.find((candidate) => candidate.glob === value);
    if (!group) {
      pickFile(value);
      return;
    }
    fileField.onChange(group.firstPath);
    batchGlobField.onChange(group.glob);
  };

  const noDatasetFiles = !!fileset && !filesQuery.isLoading && fileItems.length === 0;
  const placeholder = filesQuery.isLoading || loading ? 'Loading files...' : 'Select a file';

  return (
    <Stack gap="density-sm">
      <FilesetSearchableSelect<T>
        workspace={workspace}
        useControllerProps={{ control, name: filesetName }}
        formFieldProps={{ slotLabel: 'Fileset' }}
        renderOption={filesetOption}
        onChange={() => pickFile('')}
        disabled={disabled}
      />
      <FormField
        slotLabel="File"
        slotHelp={
          noDatasetFiles
            ? 'This fileset has no JSONL, JSON, CSV, or Parquet files.'
            : batchGlobField.value
              ? 'Read from this fileset rather than copied into the run, so later changes to these files also change re-runs.'
              : 'JSONL, JSON, CSV, or Parquet.'
        }
        slotError={error}
        status={error ? 'error' : undefined}
      >
        <Select
          disabled={disabled || !fileset}
          items={fileItems}
          value={batchGlobField.value || fileField.value}
          onValueChange={pickItem}
          renderValue={renderFileValue}
          placeholder={placeholder}
        />
      </FormField>
    </Stack>
  );
}
