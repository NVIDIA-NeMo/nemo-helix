// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvaluationFormValues,
  isSupportedMappingPath,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import {
  lastExchange,
  useDatasetPreview,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useEffect } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * Binds an OpenAI messages array, and the positional turns inside it, into
 * `fieldMapping`. The templates resolve those turns positionally, so there is
 * nothing for the user to decide; the assistant turn is recorded as the
 * reference so validation can tell a conversation apart from a prompts-only file.
 *
 * A hook rather than an effect in the Dataset panel because `toFieldMapping`
 * drops array paths before submit: a saved configuration has no `input` or
 * `reference` to restore, so the re-use path has to re-derive them.
 */
export function useMessagesBinding(): void {
  const { control, setValue } = useFormContext<EvaluationFormValues>();
  const dataset = useWatch({ control, name: 'dataset' });
  const fieldMapping = useWatch({ control, name: 'fieldMapping' });
  // Row 0 on purpose: key extraction describes the file's shape, not whichever
  // row the Live Test is pointed at.
  const { row, messagesColumn, messageSelectors } = useDatasetPreview(dataset ?? null);

  const exchange = lastExchange(messageSelectors);
  const assistantSelector = exchange.assistant ?? '';
  const userSelector = exchange.user ?? '';

  useEffect(() => {
    if (!row) return;
    const boundMessages = fieldMapping?.messages ?? '';
    const nextMessages = messagesColumn ?? '';
    if (boundMessages !== nextMessages) setValue('fieldMapping.messages', nextMessages);

    const boundReference = fieldMapping?.reference ?? '';
    const boundInput = fieldMapping?.input ?? '';
    if (messagesColumn) {
      if (boundReference !== assistantSelector)
        setValue('fieldMapping.reference', assistantSelector);
      if (boundInput !== userSelector) setValue('fieldMapping.input', userSelector);
    } else {
      // Left over from a messages dataset; a flat file cannot use array paths.
      if (boundReference && !isSupportedMappingPath(boundReference))
        setValue('fieldMapping.reference', '');
      if (boundInput && !isSupportedMappingPath(boundInput)) setValue('fieldMapping.input', '');
    }
  }, [row, messagesColumn, assistantSelector, userSelector, fieldMapping, setValue]);
}
