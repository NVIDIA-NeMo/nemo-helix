// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Checkbox, FormField, Select, Stack, Text } from '@nvidia/foundations-react-core';
import { parquetBatchGlob } from '@studio/components/evaluation/shared/parquetBatchGlob';
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
  /** Form field holding a glob over every Parquet batch beside the chosen file, or '' to read
   *  only that file. */
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
  const batches = fileField.value ? parquetBatchGlob(fileField.value, filesetPaths) : null;

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
        onChange={() => pickFile('')}
        disabled={disabled}
      />
      <FormField
        slotLabel="File"
        slotHelp={
          noDatasetFiles
            ? 'This fileset has no JSONL, JSON, CSV, or Parquet files.'
            : batches
              ? 'JSONL, JSON, CSV, or Parquet.'
              : 'JSONL, JSON, CSV, or Parquet. If the output is split across several files, only the file you pick is evaluated.'
        }
        slotError={error}
        status={error ? 'error' : undefined}
      >
        <Select
          disabled={disabled || !fileset}
          items={fileItems}
          value={fileField.value}
          onValueChange={pickFile}
          placeholder={placeholder}
        />
      </FormField>
      {batches ? (
        <Stack gap="density-xs">
          <Checkbox
            checked={batchGlobField.value === batches.glob}
            onCheckedChange={(checked) =>
              batchGlobField.onChange(checked === true ? batches.glob : '')
            }
            disabled={disabled}
            slotLabel={`Evaluate all ${batches.count} Parquet files in ${batches.dir || 'the fileset root'}`}
          />
          <Text className="text-secondary" kind="body/regular/xs">
            They are read from this fileset rather than copied into the run, so later changes to
            these files also change re-runs.
          </Text>
        </Stack>
      ) : null}
    </Stack>
  );
}
