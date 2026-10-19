// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { ControlledTextArea } from '@nemo/common/src/components/form/ControlledTextArea';
import { ControlledTextInput } from '@nemo/common/src/components/form/ControlledTextInput';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  MAX_COLUMN_INTROSPECTION_BYTES,
  SOURCE_TYPE_DATASET,
} from '@studio/routes/AnonymizerBuilderRoute/constants';
import type { AnonymizerFormData } from '@studio/routes/AnonymizerBuilderRoute/schema';
import { getContentColumns, getFileExtension } from '@studio/util/files';
import { FC, useEffect, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export const ColumnsSection: FC = () => {
  const { control, setValue } = useFormContext<AnonymizerFormData>();
  const workspace = useWorkspaceFromPath();
  const source = useWatch({ control, name: 'source' });
  const sourceType = useWatch({ control, name: 'sourceType' });
  const textColumn = useWatch({ control, name: 'textColumn' });

  const parsed = useMemo(
    () =>
      sourceType === SOURCE_TYPE_DATASET && source ? parseFilesetLocation(source, workspace) : null,
    [sourceType, source, workspace]
  );
  const filesetWorkspace = parsed?.workspace ?? '';
  const filesetName = parsed?.name ?? '';
  const filePath = parsed?.objectPath ?? '';

  const { data: filesResponse } = useFilesListFilesetFiles(
    filesetWorkspace,
    filesetName,
    undefined,
    {
      query: { enabled: Boolean(filesetWorkspace && filesetName) },
    }
  );
  const fileSize = useMemo(
    () => filesResponse?.data?.find((file) => file.path === filePath)?.size ?? null,
    [filesResponse?.data, filePath]
  );
  const tooLarge = fileSize != null && fileSize > MAX_COLUMN_INTROSPECTION_BYTES;

  const isParquet = filePath.endsWith('parquet');
  const canIntrospect = Boolean(filesetWorkspace && filesetName && filePath) && !tooLarge;
  const { data: fileContent } = useDatasetFileContent({
    workspace: filesetWorkspace,
    name: filesetName,
    path: filePath,
    range: isParquet ? [0, 1] : undefined,
    enabled: canIntrospect,
  });

  const columns = useMemo(() => {
    if (!fileContent) return [];
    const fileType = isParquet ? 'jsonl' : (getFileExtension(filePath) ?? undefined);
    return getContentColumns(fileContent, fileType);
  }, [fileContent, filePath, isParquet]);

  const columnItems = useMemo(
    () => columns.map((column) => ({ label: column, value: column })),
    [columns]
  );
  const useColumnDropdown = canIntrospect && columns.length > 0;

  useEffect(() => {
    if (useColumnDropdown && columns.length === 1 && textColumn !== columns[0]) {
      setValue('textColumn', columns[0], { shouldValidate: true });
    }
  }, [useColumnDropdown, columns, textColumn, setValue]);

  return (
    <Stack gap="density-lg">
      <Text kind="label/bold/lg">Columns</Text>
      {useColumnDropdown ? (
        <ControlledSelect
          aria-label="Text column"
          items={columnItems}
          useControllerProps={{ name: 'textColumn', control }}
          formFieldProps={{
            slotLabel: 'Text Column',
            slotInfo: 'The column containing the text to anonymize.',
          }}
        />
      ) : (
        <ControlledTextInput
          useControllerProps={{ name: 'textColumn', control }}
          placeholder="e.g. biography"
          formFieldProps={{
            slotLabel: 'Text Column',
            slotInfo: 'The column containing the text to anonymize.',
          }}
        />
      )}
      <ControlledTextArea
        useControllerProps={{ name: 'dataSummary', control }}
        formFieldProps={{
          slotLabel: 'Data Summary',
          slotInfo: 'Optional short description of the data. Helps the LLM produce better results.',
        }}
      />
    </Stack>
  );
};
