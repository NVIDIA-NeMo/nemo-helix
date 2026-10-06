// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { FormField, Select, Stack } from '@nvidia/foundations-react-core';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { formatFromFileName } from '@studio/components/FileRowEditor/parse';
import { useQueryClient } from '@tanstack/react-query';
import { type ReactElement, useRef, useState } from 'react';
import { type FieldValues, type UseControllerProps, useWatch } from 'react-hook-form';

const DATASET_FORMATS = ['json', 'jsonl', 'parquet'];

/** Parquet is decoded to JSONL text on read, so the picked file is named for what it now holds. */
const pickedFileName = (path: string): string => {
  const baseName = path.split('/').pop() ?? path;
  return formatFromFileName(baseName) === 'parquet'
    ? baseName.replace(/\.(parquet|pq)$/i, '.jsonl')
    : baseName;
};

const filesetOption = (fileset: { name: string }) => ({
  value: fileset.name,
  label: fileset.name,
});

interface FilesetDatasetPickerProps<T extends FieldValues> {
  workspace: string;
  /** Form field holding the chosen fileset name. */
  filesetControllerProps: UseControllerProps<T>;
  disabled?: boolean;
  error?: string;
  onPick: (file: File) => void;
  onClear: () => void;
}

export function FilesetDatasetPicker<T extends FieldValues>({
  workspace,
  filesetControllerProps,
  disabled,
  error,
  onPick,
  onClear,
}: FilesetDatasetPickerProps<T>): ReactElement {
  const queryClient = useQueryClient();
  const filesetName: string = useWatch({
    control: filesetControllerProps.control,
    name: filesetControllerProps.name,
  });
  const [path, setPath] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const requestId = useRef(0);

  const filesQuery = useFilesListFilesetFiles(workspace, filesetName, undefined, {
    query: { enabled: !!filesetName },
  });

  const fileItems = (filesQuery.data?.data ?? [])
    .filter((file) => DATASET_FORMATS.includes(formatFromFileName(file.path)))
    .map((file) => ({ value: file.path, children: file.path }));

  const handleFilesetChange = () => {
    requestId.current += 1;
    setPath('');
    setLoadError(null);
    onClear();
  };

  const handleFileChange = async (value: string) => {
    requestId.current += 1;
    const current = requestId.current;
    setPath(value);
    setLoadError(null);
    onClear();
    try {
      const text = await queryClient.fetchQuery(
        datasetFileContentQueryOptions({
          workspace,
          name: filesetName,
          path: value,
          fullContent: true,
        })
      );
      if (current !== requestId.current) return;
      onPick(new File([text], pickedFileName(value)));
    } catch (err) {
      if (current !== requestId.current) return;
      setLoadError(err instanceof Error ? err.message : 'Could not read the file');
    }
  };

  const noDatasetFiles = !!filesetName && !filesQuery.isLoading && fileItems.length === 0;

  return (
    <Stack gap="density-sm">
      <FilesetSearchableSelect<T>
        workspace={workspace}
        useControllerProps={filesetControllerProps}
        formFieldProps={{ slotLabel: 'Fileset' }}
        renderOption={filesetOption}
        onChange={handleFilesetChange}
        disabled={disabled}
      />
      <FormField
        slotLabel="File"
        slotHelp={noDatasetFiles ? 'This fileset has no JSONL, JSON, or Parquet files.' : undefined}
        slotError={loadError ?? error}
        status={(loadError ?? error) ? 'error' : undefined}
      >
        <Select
          disabled={disabled || !filesetName}
          items={fileItems}
          value={path}
          onValueChange={(value) => void handleFileChange(value)}
          placeholder={filesQuery.isLoading ? 'Loading files...' : 'Select a file'}
        />
      </FormField>
    </Stack>
  );
}
