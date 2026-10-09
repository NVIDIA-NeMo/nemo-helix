// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import { parseFilesetLocation } from '@nemo/common/src/components/DatasetFileSelect/parseFilesetLocation';
import { DatasetRowPager } from '@nemo/common/src/components/DatasetRowPager';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { PreviewBox } from '@nemo/common/src/components/PreviewBox';
import { resolveKeyPath } from '@nemo/common/src/utils/file';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Banner, Block, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  CANONICAL_FIELD_LABELS,
  type CanonicalField,
  EMPTY_FIELD_MAPPING,
  type EvaluationFormValues,
  isArrayPath,
  MAPPABLE_FILE_TYPES,
  PRIMARY_CANONICAL_FIELDS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import {
  messagesShape,
  useDatasetPreview,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useMessagesBinding } from '@studio/routes/evaluation/EvaluationNewRoute/useMessagesBinding';
import { CircleCheck, CircleHelp } from 'lucide-react';
import { FC, useEffect, useMemo, useState } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

const asText = (value: unknown): string =>
  typeof value === 'string' ? value : JSON.stringify(value);

const Check: FC<{ ok: boolean; label: string }> = ({ ok, label }) => (
  <Flex align="center" gap="density-sm">
    {ok ? (
      <CircleCheck className="text-feedback-success shrink-0" width={16} height={16} />
    ) : (
      <CircleHelp className="text-secondary shrink-0" width={16} height={16} />
    )}
    <Text kind="body/regular/md">{label}</Text>
  </Flex>
);

export const DatasetPanel: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { control, setError, clearErrors, setValue } = useFormContext<EvaluationFormValues>();
  const dataset = useWatch({ control, name: 'dataset' });
  const fieldMapping = useWatch({ control, name: 'fieldMapping' });
  // Row 0 on purpose: key extraction describes the file's shape, not whichever
  // row the Live Test is pointed at.
  const { row, keyOptions, messagesColumn, messageSelectors, isLoading, error } = useDatasetPreview(
    dataset ?? null
  );

  useMessagesBinding();

  const [rowIndex, setRowIndex] = useState(0);
  useEffect(() => setRowIndex(0), [dataset]);
  const { row: previewRow, rowCount, isPartial } = useDatasetPreview(dataset ?? null, rowIndex);
  const fileName = dataset ? parseFilesetLocation(dataset)?.objectPath.split('/').pop() : null;

  const shape = messagesShape(messageSelectors);

  const formatLabel = (dataset?.split('.').pop() ?? '').toUpperCase() || 'File';

  const bindableOptions = useMemo(
    () => keyOptions.filter((option) => !isArrayPath(option.value)),
    [keyOptions]
  );

  const hasIngestedKeys = bindableOptions.length > 0;

  /** Select renders an item's ``children``; a ``label`` key is ignored and the
   *  raw value shows through instead. */
  const items = useMemo(
    () => bindableOptions.map((option) => ({ value: option.value, children: option.label })),
    [bindableOptions]
  );

  const renderMappingSelect = (field: CanonicalField) => (
    <Block key={field} className="min-w-0 flex-1">
      <ControlledSelect
        useControllerProps={{ name: `fieldMapping.${field}` as const, control }}
        formFieldProps={{ slotLabel: CANONICAL_FIELD_LABELS[field] }}
        items={items}
        loading={isLoading}
        dismissible
        placeholder={`Select a field for ${CANONICAL_FIELD_LABELS[field]}`}
      />
    </Block>
  );

  const renderMappingPreview = (field: CanonicalField) => {
    const path = fieldMapping?.[field];
    const value = previewRow && path ? resolveKeyPath(previewRow, path) : undefined;

    return (
      <Block key={field} className="min-w-0 flex-1">
        {path ? (
          <PreviewBox
            value={value === undefined || value === null ? '' : asText(value)}
            label={`${CANONICAL_FIELD_LABELS[field]} preview`}
            rows={2}
          />
        ) : null}
      </Block>
    );
  };

  const anyMapped = PRIMARY_CANONICAL_FIELDS.some((field) => fieldMapping?.[field]);

  return (
    <Stack justify="start" gap="density-2xl">
      <Stack gap="density-lg">
        <ControlledDatasetFileSelect
          label="Input File"
          workspace={workspace}
          useControllerProps={{ name: 'dataset', control }}
          setError={(fieldError) => setError('dataset', fieldError)}
          clearError={() => clearErrors('dataset')}
          acceptedFileTypes={[...MAPPABLE_FILE_TYPES]}
          invalidFileMode="disable"
          filesetPurpose={FilesetPurpose.dataset}
          autoSelectFirstAcceptable
          onFileSelected={() => setValue('fieldMapping', EMPTY_FIELD_MAPPING)}
        />

        {error ? (
          <Banner kind="inline" status="error">
            {error}
          </Banner>
        ) : null}

        {row ? (
          <Stack gap="density-sm" className="rounded-md border border-base p-density-lg">
            <Text kind="body/bold/lg">File Validation</Text>
            <Check ok label={`${formatLabel} is valid`} />
            {messagesColumn ? (
              <>
                <Check ok label={`Standard messages array found in "${messagesColumn}"`} />
                <Check
                  ok={shape.input}
                  label={
                    shape.input
                      ? 'Input mapped to the last user message'
                      : 'No user message to use as Input'
                  }
                />
                <Check
                  ok={shape.reference}
                  label={
                    shape.reference
                      ? 'Reference mapped to the last assistant message'
                      : 'No assistant reply to the last user message to use as Reference'
                  }
                />
              </>
            ) : (
              <Check ok={false} label="Map the Input and Reference fields below" />
            )}
          </Stack>
        ) : null}

        {/* Mapping is only for ambiguity. An OpenAI messages array has none: the
            last user turn is the input and the last assistant turn is the ground
            truth, bound by role in useMessagesBinding. Asking the user to map
            that would be busywork. */}
        {hasIngestedKeys && !messagesColumn ? (
          <Stack gap="density-sm" className="min-w-0">
            <Text kind="body/bold/lg">Field Mapping</Text>
            <Text kind="body/regular/md" className="text-secondary">
              Choose which field holds the prompt to send to the model, and which holds the answer
              to score its response against. Reference is only needed by metrics that compare
              against it.
            </Text>
            <Flex align="start" gap="density-lg" className="min-w-0">
              {PRIMARY_CANONICAL_FIELDS.map(renderMappingSelect)}
            </Flex>

            <Flex align="start" gap="density-lg" className="min-w-0">
              {PRIMARY_CANONICAL_FIELDS.map(renderMappingPreview)}
            </Flex>

            {anyMapped && rowCount > 1 ? (
              <DatasetRowPager
                fileName={fileName}
                rowIndex={rowIndex}
                rowCount={rowCount}
                isPartial={isPartial}
                onChange={setRowIndex}
              />
            ) : null}
          </Stack>
        ) : null}
      </Stack>
    </Stack>
  );
};
