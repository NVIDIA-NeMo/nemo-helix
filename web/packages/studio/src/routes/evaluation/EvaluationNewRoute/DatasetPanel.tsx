// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ControlledDatasetFileSelect } from '@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect';
import { ControlledSelect } from '@nemo/common/src/components/form/ControlledSelect';
import { FilesetPurpose } from '@nemo/sdk/generated/platform/schema';
import { Banner, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  CANONICAL_FIELD_LABELS,
  type CanonicalField,
  type EvaluationFormValues,
  isSupportedMappingPath,
  MAPPABLE_FILE_TYPES,
  PRIMARY_CANONICAL_FIELDS,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetPreview } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { CircleCheck, CircleHelp } from 'lucide-react';
import { FC, useEffect, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

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
  // row the Dry Run is pointed at.
  const { row, keyOptions, messagesColumn, isLoading, error } = useDatasetPreview(dataset ?? null);

  const formatLabel = (dataset?.split('.').pop() ?? '').toUpperCase() || 'File';

  /** An OpenAI messages array binds as a whole column, so bind it automatically:
   *  there is nothing for the user to decide, and the templates resolve the user
   *  and assistant turns positionally. */
  useEffect(() => {
    if (!row) return;
    const bound = fieldMapping?.messages ?? '';
    const next = messagesColumn ?? '';
    if (bound !== next) setValue('fieldMapping.messages', next);
  }, [row, messagesColumn, fieldMapping, setValue]);

  const bindableOptions = useMemo(
    () => keyOptions.filter((option) => isSupportedMappingPath(option.value)),
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
    <ControlledSelect
      key={field}
      useControllerProps={{ name: `fieldMapping.${field}` as const, control }}
      formFieldProps={{ slotLabel: CANONICAL_FIELD_LABELS[field] }}
      items={items}
      loading={isLoading}
      dismissible
      placeholder={`Select a column for ${CANONICAL_FIELD_LABELS[field]}`}
    />
  );

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
                <Check ok label="Input mapped to the user message" />
                <Check ok label="Ground Truth mapped to the assistant message" />
              </>
            ) : (
              <Check ok={false} label="Assign data fields to metrics below:" />
            )}
          </Stack>
        ) : null}

        {/* Mapping is only for ambiguity. An OpenAI messages array has none: the
            user turn is the input and the assistant turn is the ground truth, so
            the whole column binds to canonical `messages` and the templates index
            it positionally. Asking the user to map that would be busywork. */}
        {hasIngestedKeys && !messagesColumn ? (
          <Stack gap="density-lg">{PRIMARY_CANONICAL_FIELDS.map(renderMappingSelect)}</Stack>
        ) : null}
      </Stack>
    </Stack>
  );
};
