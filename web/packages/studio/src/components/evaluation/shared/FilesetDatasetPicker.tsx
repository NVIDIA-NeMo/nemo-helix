// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesListFilesetFiles, useFilesListFilesets } from '@nemo/sdk/generated/platform/files';
import { FormField, Select, Stack } from '@nvidia/foundations-react-core';
import { datasetFileContentQueryOptions } from '@studio/api/datasets/useDatasetFileContent';
import { formatFromFileName } from '@studio/components/FileRowEditor/parse';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useQueryClient } from '@tanstack/react-query';
import { type FC, useRef, useState } from 'react';

const DATASET_FORMATS = ['json', 'jsonl', 'parquet'];

/** Parquet is decoded to JSONL text on read, so the picked file is named for what it now holds. */
const pickedFileName = (path: string): string => {
  const baseName = path.split('/').pop() ?? path;
  return formatFromFileName(baseName) === 'parquet'
    ? baseName.replace(/\.(parquet|pq)$/i, '.jsonl')
    : baseName;
};

interface FilesetDatasetPickerProps {
  workspace: string;
  disabled?: boolean;
  error?: string;
  onPick: (file: File) => void;
  onClear: () => void;
}

export const FilesetDatasetPicker: FC<FilesetDatasetPickerProps> = ({
  workspace,
  disabled,
  error,
  onPick,
  onClear,
}) => {
  const queryClient = useQueryClient();
  const [filesetName, setFilesetName] = useState('');
  const [path, setPath] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const requestId = useRef(0);

  const filesetsQuery = useFilesListFilesets(workspace, { page_size: DEFAULT_LARGE_PAGE_SIZE });
  const filesQuery = useFilesListFilesetFiles(workspace, filesetName, undefined, {
    query: { enabled: !!filesetName },
  });

  const filesetItems = (filesetsQuery.data?.data ?? []).flatMap((fileset) =>
    fileset.name ? [{ value: fileset.name, children: fileset.name }] : []
  );
  const fileItems = (filesQuery.data?.data ?? [])
    .filter((file) => DATASET_FORMATS.includes(formatFromFileName(file.path)))
    .map((file) => ({ value: file.path, children: file.path }));

  const handleFilesetChange = (value: string) => {
    requestId.current += 1;
    setFilesetName(value);
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
      <FormField slotLabel="Fileset">
        <Select
          disabled={disabled}
          items={filesetItems}
          value={filesetName}
          onValueChange={handleFilesetChange}
          placeholder={filesetsQuery.isLoading ? 'Loading filesets...' : 'Select a fileset'}
        />
      </FormField>
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
};
