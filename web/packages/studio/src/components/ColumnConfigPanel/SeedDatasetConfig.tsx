// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetSearchableSelect } from '@nemo/common/src/components/FilesetSearchableSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import { useFilesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { Button, Flex, FormField, Tag, Text } from '@nvidia/foundations-react-core';
import { useDatasetFileContent } from '@studio/api/datasets/useDatasetFileContent';
import { FilesetFilePreviewPanel } from '@studio/components/FilesetFilePreviewPanel';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  SAMPLING_STRATEGY_OPTIONS,
  SEED_AVAILABLE_COLUMNS_KEY,
  SEED_FILE_PATH_KEY,
  SEED_FILESET_REF_KEY,
  SEED_SAMPLING_STRATEGY_KEY,
} from '@studio/routes/DataDesignerJobBuildRoute/columns';
import type { JobBuilderFormValues } from '@studio/routes/DataDesignerJobBuildRoute/useJobBuilder';
import { getContentColumns, getFileExtension } from '@studio/util/files';
import { Eye } from 'lucide-react';
import { type FC, useEffect, useMemo, useRef, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

export interface SeedDatasetConfigProps {
  columnIndex: number;
}

/**
 * Config controls for a seed-dataset column, sourced from a platform fileset.
 *
 * The SDK's `FilesetFileSeedSource` takes a single composite `path`
 * (`{workspace}/{fileset}#{file}`); rather than have the user hand-type that, this collects the
 * fileset and the in-fileset file as separate picks (stored under {@link SEED_FILESET_REF_KEY} /
 * {@link SEED_FILE_PATH_KEY}). `buildSeedConfig` assembles them into the composite path at submit.
 *
 * Every control is registered directly on the build form. This keeps the fileset controls in the
 * same state tree as the rest of the column and isolates updates to their own subscribers.
 */
export const SeedDatasetConfig: FC<SeedDatasetConfigProps> = ({ columnIndex }) => {
  const workspace = useWorkspaceFromPath();
  const { control, setValue } = useFormContext<JobBuilderFormValues>();
  const filesetRefPath = `columns.${columnIndex}.values.${SEED_FILESET_REF_KEY}` as const;
  const filePathPath = `columns.${columnIndex}.values.${SEED_FILE_PATH_KEY}` as const;
  const samplingStrategyPath =
    `columns.${columnIndex}.values.${SEED_SAMPLING_STRATEGY_KEY}` as const;
  const availableColumnsPath =
    `columns.${columnIndex}.values.${SEED_AVAILABLE_COLUMNS_KEY}` as const;
  const filesetRef = useWatch({ control, name: filesetRefPath }) ?? '';
  const filePath = useWatch({ control, name: filePathPath }) ?? '';
  const availableColumnsValue = useWatch({
    control,
    name: availableColumnsPath,
  });

  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const previousFilesetRef = useRef(filesetRef);
  useEffect(() => {
    if (previousFilesetRef.current === filesetRef) return;
    previousFilesetRef.current = filesetRef;
    setValue(filePathPath, '');
    setValue(availableColumnsPath, '');
    setIsPreviewOpen(false);
  }, [availableColumnsPath, filePathPath, filesetRef, setValue]);

  const { workspace: filesetWorkspace, name: filesetName } = getPartsFromReference(filesetRef);
  const { data: filesResponse, isLoading: isLoadingFiles } = useFilesListFilesetFiles(
    filesetWorkspace,
    filesetName,
    undefined,
    { query: { enabled: Boolean(filesetRef) } }
  );
  const fileItems = useMemo(
    () => (filesResponse?.data ?? []).map((file) => ({ children: file.path, value: file.path })),
    [filesResponse?.data]
  );

  const isParquet = filePath.endsWith('parquet');
  const {
    data: fileContent,
    isLoading: isLoadingSchema,
    isError: isSchemaError,
  } = useDatasetFileContent({
    workspace: filesetWorkspace,
    name: filesetName,
    path: filePath,
    range: isParquet ? [0, 1] : undefined,
    enabled: Boolean(filesetRef && filePath),
  });
  const availableColumns = useMemo(() => {
    if (!fileContent) return [];
    const fileType = isParquet ? 'jsonl' : (getFileExtension(filePath) ?? undefined);
    return getContentColumns(fileContent, fileType);
  }, [fileContent, filePath, isParquet]);

  useEffect(() => {
    const joined = availableColumns.join(',');
    if (availableColumnsValue !== joined) {
      setValue(availableColumnsPath, joined);
    }
  }, [availableColumns, availableColumnsPath, availableColumnsValue, setValue]);

  const samplingItems = SAMPLING_STRATEGY_OPTIONS.map((option) => ({
    children: option.label,
    value: option.value,
  }));

  return (
    <>
      <FilesetSearchableSelect
        workspace={workspace}
        useControllerProps={{ control, name: filesetRefPath }}
        formFieldProps={{
          slotLabel: 'Fileset',
          slotInfo: 'The platform fileset to seed rows from.',
        }}
        triggerPlaceholder="Select a fileset"
      />

      <ControlledSelect
        aria-label="Seed file"
        disabled={!filesetRef}
        items={fileItems}
        useControllerProps={{ name: filePathPath }}
        formFieldProps={{
          slotLabel: (
            <Flex align="center" gap="density-xxs">
              File
              <Button
                kind="tertiary"
                color="neutral"
                size="small"
                aria-label="Preview seed file"
                disabled={!filePath}
                onClick={() => setIsPreviewOpen(true)}
                className="h-auto p-0"
              >
                <Eye size={14} aria-hidden />
              </Button>
            </Flex>
          ),
          required: true,
          slotInfo: 'The file within the fileset to read rows from.',
        }}
        onChange={() => {
          setValue(availableColumnsPath, '');
          setIsPreviewOpen(false);
        }}
        placeholder={
          !filesetRef
            ? 'Select a fileset first'
            : isLoadingFiles
              ? 'Loading files…'
              : 'Select a file'
        }
      />

      {filePath && (
        <FormField
          slotLabel="Available columns"
          slotInfo="Columns provided by the seed file. Reference them from other columns with {{ name }}."
        >
          {isLoadingSchema ? (
            <Text kind="body/regular/sm" className="text-secondary">
              Reading columns…
            </Text>
          ) : isSchemaError ? (
            <Text kind="body/regular/sm" className="text-feedback-danger">
              Couldn't read columns from this file.
            </Text>
          ) : availableColumns.length === 0 ? (
            <Text kind="body/regular/sm" className="text-secondary">
              No columns found in this file.
            </Text>
          ) : (
            <Flex gap="density-xs" className="flex-wrap">
              {availableColumns.map((name) => (
                <Tag key={name} kind="outline" color="gray" readOnly>
                  {name}
                </Tag>
              ))}
            </Flex>
          )}
        </FormField>
      )}

      <ControlledSelect
        aria-label="Sampling strategy"
        items={samplingItems}
        useControllerProps={{ name: samplingStrategyPath }}
        formFieldProps={{
          slotLabel: 'Sampling strategy',
          slotInfo: 'How rows are read from the seed dataset. Defaults to ordered.',
        }}
        placeholder="Ordered"
      />

      <FilesetFilePreviewPanel
        open={isPreviewOpen}
        onCloseClick={() => setIsPreviewOpen(false)}
        onOutsideClick={() => setIsPreviewOpen(false)}
        workspace={filesetWorkspace}
        filesetName={filesetName}
        filePath={filePath}
      />
    </>
  );
};
