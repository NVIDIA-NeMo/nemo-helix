// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Block, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { AutoSplitNotice } from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel/AutoSplitNotice';
import {
  ChecklistRow,
  type ChecklistStatus,
} from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel/ChecklistRow';
import { SchemaBlock } from '@studio/components/customizer/CustomizationFilesetSelect/FileValidationPanel/SchemaBlock';
import type { CustomizationDatasetValidationResult } from '@studio/hooks/useCustomizationDatasetValidation';
import type { FC } from 'react';

export interface FileValidationPanelProps {
  validation: CustomizationDatasetValidationResult;
}

export const FileValidationPanel: FC<FileValidationPanelProps> = ({ validation }) => {
  const {
    isPending,
    discoveryError,
    format,
    schema,
    schemaExpectedCopy,
    schemaMismatchedFiles,
    schemaShape,
    completeness,
    encoding,
    hasTraining,
    autoSplitNotice,
    trainingRowCount,
  } = validation;

  // Discovery (file-listing) failure is distinct from "the dataset is empty".
  // Render an explicit load error so users don't think their data is bad.
  if (discoveryError) {
    return (
      <Banner kind="inline" status="error">
        Could not list files in this dataset: {discoveryError.message || 'unknown error'}. Pick the
        dataset again or try once more.
      </Banner>
    );
  }

  if (isPending) {
    return (
      <Flex align="center" gap="density-sm" className="py-density-md">
        <Spinner size="small" description="Validating dataset files..." />
      </Flex>
    );
  }

  // No-training error supersedes the rest of the checklist.
  if (!hasTraining) {
    return null;
  }

  const formatStatus: ChecklistStatus = format.ok ? 'ok' : 'fail';
  const formatLabel = format.ok ? (
    'Single line JSONL is valid'
  ) : format.fileErrors.length === 0 ? (
    'Format check failed'
  ) : (
    <>
      Found {format.fileErrors.length} file
      {format.fileErrors.length === 1 ? '' : 's'} with errors (e.g. {format.fileErrors[0].path} —{' '}
      {format.fileErrors[0].error})
    </>
  );

  // Schema check fails when:
  //   - no file matched a recognized customizer shape (warning), OR
  //   - some files matched but others didn't / used a different variant (fail).
  // Customizer would still reject the latter at training time, so the panel
  // can't silently treat the first match as authoritative.
  const schemaHasMismatches = schemaMismatchedFiles.length > 0;
  const schemaStatus: ChecklistStatus = !schema ? 'warning' : schemaHasMismatches ? 'fail' : 'ok';
  const schemaLabel = !schema ? (
    <>Does not match. {schemaExpectedCopy}</>
  ) : schemaHasMismatches ? (
    <>
      {schema.label}, but {schemaMismatchedFiles.length} file
      {schemaMismatchedFiles.length === 1 ? '' : 's'} do not match (e.g. {schemaMismatchedFiles[0]}
      ).
    </>
  ) : (
    <>{schema.label}</>
  );

  return (
    <Stack gap="density-lg">
      {autoSplitNotice && <AutoSplitNotice trainingRowCount={trainingRowCount} />}

      {/* File Validation + Schema preview share a subdued grouped surface
          per design spec — visually distinct from the auto-split notice
          (which lives on the form's main surface). bg-surface-sunken matches
          the recessed look used elsewhere in the app for inset content. */}
      <Block className="bg-surface-sunken rounded-lg" padding="density-lg">
        <Stack gap="density-lg">
          <Stack gap="density-sm">
            <Text kind="label/bold/sm">File Validation</Text>
            <ChecklistRow status={formatStatus} label={<>Format: {formatLabel}</>} />
            <ChecklistRow status={schemaStatus} label={<>Schema: {schemaLabel}</>} />
            <ChecklistRow
              status={encoding.ok ? 'ok' : 'fail'}
              label={
                <>
                  Encoding:{' '}
                  {encoding.ok
                    ? 'UTF-8 encoding (not UTF-16 or other encodings)'
                    : encoding.fileErrors.length === 0
                      ? 'UTF-8 encoding check failed'
                      : encoding.fileErrors.length === 1
                        ? `${encoding.fileErrors[0].path} is not valid UTF-8`
                        : `${encoding.fileErrors.length} files are not valid UTF-8 (e.g. ${encoding.fileErrors[0].path})`}
                </>
              }
            />
            {/* Completeness only renders when it could be evaluated. When
                format failed or schema didn't match (skipped=true) we don't
                know what to require, so hiding the row is honest — better
                than a third "neutral" state. */}
            {!completeness.skipped && (
              <ChecklistRow
                status={completeness.ok ? 'ok' : 'fail'}
                label={
                  <>
                    Completeness:{' '}
                    {completeness.ok
                      ? 'No empty or null values in required fields'
                      : completeness.errors.length === 0
                        ? 'Completeness check failed'
                        : `Found ${completeness.errors.length} row${
                            completeness.errors.length === 1 ? '' : 's'
                          } with empty or missing required fields (e.g. ${completeness.errors[0].path}:${completeness.errors[0].row} — ${completeness.errors[0].message})`}
                  </>
                }
              />
            )}
            {/* Length is intentionally not rendered here — we don't run a
                tokenizer client-side. Customizer enforces context-window limits
                at training time; surfacing a row we can't actually evaluate
                would be misleading. */}
            {/* Naming was previously rendered here ("Training and validation
                sets are separate and representative") but its check (namingOk
                = hasTraining) was redundant with the no-training error banner
                above and the label promised verifications we don't and can't
                run. Dropped to keep the panel honest. */}
          </Stack>

          <SchemaBlock schemaShape={schemaShape} />
        </Stack>
      </Block>
    </Stack>
  );
};
