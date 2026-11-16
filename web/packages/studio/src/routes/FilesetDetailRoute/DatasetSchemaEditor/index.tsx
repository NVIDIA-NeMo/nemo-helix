// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CodeEditor } from '@nemo/common/src/components/CodeEditor';
import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import { SUPPORTED_FILE_FORMATS } from '@nemo/common/src/types';
import type { FilesetFileOutput, FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, Stack, TableToolbar, Text } from '@nvidia/foundations-react-core';
import { SchemaSelectControl } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/SchemaSelectControl';
import { SharedSchemaConfirmModal } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/SharedSchemaConfirmModal';
import { useDatasetSchemaEditor } from '@studio/routes/FilesetDetailRoute/DatasetSchemaEditor/useDatasetSchemaEditor';
import type { FC } from 'react';

export interface DatasetSchemaEditorProps {
  workspace: string;
  datasetName: string;
  fileset: FilesetOutput;
  filesList: FilesetFileOutput[] | undefined;
  /** When set, dropdown auto-selects the file's mapped schema. When cleared,
   *  dropdown jumps back to "Show All". */
  selectedFilePath?: string;
}

/**
 * Dataset-specific schema editor orchestrator.
 *
 * Renders a dropdown ("Default" + each `schema_defs` key + "Show All
 * advanced"). The editor shows one JSON Schema at a time in single-schema
 * mode, or the full `metadata.dataset` payload when "Show All" is picked.
 *
 * User edits are preserved per-selection so switching schemas and back
 * doesn't lose work.
 */
export const DatasetSchemaEditor: FC<DatasetSchemaEditorProps> = ({
  workspace,
  datasetName,
  fileset,
  filesList,
  selectedFilePath,
}) => {
  const {
    defKeys,
    defaultDefKey,
    hasInlineDefault,
    selectedSchema,
    setSelectedSchema,
    text,
    userEdited,
    handleEditorChange,
    handleReset,
    isInferring,
    inferError,
    supportedExistingFiles,
    handleInferFromExisting,
    isSaving,
    validation,
    sharedReferrerCount,
    pendingShareConfirm,
    setPendingShareConfirm,
    handleSave,
    handleConfirmShared,
    handleSetDefault,
    canInferFromExisting,
    canSave,
    canSetDefault,
    isEmpty,
  } = useDatasetSchemaEditor({
    workspace,
    datasetName,
    fileset,
    filesList,
    selectedFilePath,
  });

  if (isEmpty) {
    return (
      <Stack
        gap="density-md"
        align="center"
        className="w-full py-density-2xl px-density-md text-center"
        data-testid="dataset-schema-editor-empty"
      >
        <Text kind="title/sm">No schema yet</Text>
        <Text kind="body/regular/sm">
          This dataset doesn&apos;t have a schema set. A schema helps downstream tools (training,
          evaluation) validate your file shapes.
        </Text>
        <Button
          kind="primary"
          onClick={handleInferFromExisting}
          disabled={!canInferFromExisting}
          data-testid="dataset-schema-infer-button"
        >
          {isInferring ? 'Generating…' : 'Generate'}
        </Button>
        {supportedExistingFiles.length === 0 && (
          <Text kind="body/regular/sm">
            Upload a supported file ({SUPPORTED_FILE_FORMATS.map((f) => `.${f}`).join(', ')}) to
            enable auto-inference.
          </Text>
        )}
        {inferError && (
          <Text kind="body/regular/sm" data-testid="dataset-schema-infer-error">
            {inferError}
          </Text>
        )}
      </Stack>
    );
  }

  return (
    <Stack gap="density-md" className="w-full h-full min-h-0" data-testid="dataset-schema-editor">
      <TableToolbar aria-label="Dataset schema toolbar" className="min-w-0 shrink-0">
        <Flex direction="row" gap="density-md" className="min-w-0 w-full">
          <Flex align="center" className="flex-1 min-w-0">
            <Text kind="title/sm">Dataset Schema</Text>
          </Flex>
          <Button
            kind="secondary"
            onClick={handleInferFromExisting}
            disabled={!canInferFromExisting}
            data-testid="dataset-schema-infer-button"
          >
            {isInferring ? 'Generating…' : 'Generate'}
          </Button>
        </Flex>
      </TableToolbar>

      <Flex
        gap="density-sm"
        align="center"
        className="shrink-0"
        data-testid="dataset-schema-select-row"
      >
        <div className="flex-1 min-w-0">
          <SchemaSelectControl
            defKeys={defKeys}
            hasInlineDefault={hasInlineDefault}
            defaultDefKey={defaultDefKey}
            value={selectedSchema}
            onChange={setSelectedSchema}
          />
        </div>
        <Button
          kind="secondary"
          onClick={handleSetDefault}
          disabled={!canSetDefault}
          data-testid="dataset-schema-set-default-button"
        >
          Set Default
        </Button>
      </Flex>

      <CodeEditor
        content={text}
        contentType={ContentType.JSON}
        onChange={handleEditorChange}
        className="flex-1 min-h-0 overflow-hidden"
        hideCopyButton
      />

      {!validation.valid && validation.errors.length > 0 && (
        <Stack gap="density-xs" className="shrink-0" data-testid="dataset-schema-editor-errors">
          {validation.errors.map((err, i) => (
            <Text key={i} kind="body/regular/sm">
              {err}
            </Text>
          ))}
        </Stack>
      )}

      {inferError && (
        <Text kind="body/regular/sm" className="shrink-0" data-testid="dataset-schema-infer-error">
          {inferError}
        </Text>
      )}

      <Flex gap="density-md" justify="end" className="shrink-0">
        <Button
          kind="secondary"
          onClick={handleReset}
          disabled={!userEdited}
          data-testid="dataset-schema-reset-button"
        >
          Reset
        </Button>
        <Button
          kind="primary"
          color="brand"
          onClick={handleSave}
          disabled={!canSave}
          data-testid="dataset-schema-save-button"
        >
          {isSaving ? 'Saving…' : 'Save schema'}
        </Button>
      </Flex>

      <SharedSchemaConfirmModal
        open={pendingShareConfirm}
        referrerCount={sharedReferrerCount}
        isPending={isSaving}
        onConfirm={handleConfirmShared}
        onCancel={() => setPendingShareConfirm(false)}
      />
    </Stack>
  );
};
