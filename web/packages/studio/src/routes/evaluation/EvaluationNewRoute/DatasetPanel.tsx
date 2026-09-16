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
import {
  lastSelectorForRole,
  useDatasetPreview,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
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
  // row the Live Test is pointed at.
  const { row, keyOptions, messagesColumn, messageSelectors, isLoading, error } = useDatasetPreview(
    dataset ?? null
  );

  const assistantSelector = lastSelectorForRole(messageSelectors, 'assistant') ?? '';
  const userSelector = lastSelectorForRole(messageSelectors, 'user') ?? '';

  const formatLabel = (dataset?.split('.').pop() ?? '').toUpperCase() || 'File';

  /** An OpenAI messages array binds as a whole column, so bind it automatically:
   *  there is nothing for the user to decide, and the templates resolve the user
   *  and assistant turns positionally.
   *
   *  The assistant turn is recorded as the reference so validation can tell a
   *  conversation apart from a prompts-only file. ``toFieldMapping`` drops it
   *  before submit, because an array path is not a legal column mapping. */
  useEffect(() => {
    if (!row) return;
    const boundMessages = fieldMapping?.messages ?? '';
    const nextMessages = messagesColumn ?? '';
    if (boundMessages !== nextMessages) setValue('fieldMapping.messages', nextMessages);

    const boundReference = fieldMapping?.reference ?? '';
    if (messagesColumn) {
      if (boundReference !== assistantSelector)
        setValue('fieldMapping.reference', assistantSelector);
    } else if (boundReference && !isSupportedMappingPath(boundReference)) {
      // Left over from a messages dataset; a flat file cannot use it.
      setValue('fieldMapping.reference', '');
    }
  }, [row, messagesColumn, assistantSelector, fieldMapping, setValue]);

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
                <Check ok={Boolean(userSelector)} label="Input mapped to the user message" />
                <Check
                  ok={Boolean(assistantSelector)}
                  label={
                    assistantSelector
                      ? 'Ground Truth mapped to the assistant message'
                      : 'No assistant message to use as Ground Truth'
                  }
                />
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
