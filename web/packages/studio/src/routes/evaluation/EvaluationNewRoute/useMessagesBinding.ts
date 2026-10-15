// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvaluationFormValues,
  isArrayPath,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import {
  messagesPath,
  messagesShape,
  useDatasetPreview,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useEffect } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * Binds an OpenAI messages array into `fieldMapping`: Input to the last user
 * turn and Reference to the last assistant turn, by role so every row resolves
 * its own turns. There is nothing for the user to decide; the Reference is only
 * bound when row 0 has a reply to pair with, so validation can tell a
 * conversation apart from a prompts-only file.
 *
 * `messages` stays bound to the whole column so configurations saved before
 * role paths existed, whose judge prompts index `{{ messages[n].content }}`,
 * still render.
 *
 * A hook rather than an effect in the Dataset panel because the re-use path
 * needs it too: re-deriving is idempotent for a current saved configuration,
 * and fills in Input and Reference for an older one that never carried them.
 */
export function useMessagesBinding(): void {
  const { control, setValue } = useFormContext<EvaluationFormValues>();
  const dataset = useWatch({ control, name: 'dataset' });
  const fieldMapping = useWatch({ control, name: 'fieldMapping' });
  // Row 0 on purpose: key extraction describes the file's shape, not whichever
  // row the Live Test is pointed at.
  const { row, messagesColumn, messageSelectors } = useDatasetPreview(dataset ?? null);

  const shape = messagesShape(messageSelectors);
  const userPath = messagesColumn && shape.input ? messagesPath(messagesColumn, 'user') : '';
  const assistantPath =
    messagesColumn && shape.reference ? messagesPath(messagesColumn, 'assistant') : '';

  useEffect(() => {
    if (!row) return;
    const boundMessages = fieldMapping?.messages ?? '';
    const nextMessages = messagesColumn ?? '';
    if (boundMessages !== nextMessages) setValue('fieldMapping.messages', nextMessages);

    const boundReference = fieldMapping?.reference ?? '';
    const boundInput = fieldMapping?.input ?? '';
    if (messagesColumn) {
      if (boundReference !== assistantPath) setValue('fieldMapping.reference', assistantPath);
      if (boundInput !== userPath) setValue('fieldMapping.input', userPath);
    } else {
      // Left over from a messages dataset; a flat file binds columns, not turns.
      if (isArrayPath(boundReference)) setValue('fieldMapping.reference', '');
      if (isArrayPath(boundInput)) setValue('fieldMapping.input', '');
    }
  }, [row, messagesColumn, assistantPath, userPath, fieldMapping, setValue]);
}
