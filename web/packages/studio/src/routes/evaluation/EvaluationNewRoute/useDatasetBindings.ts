// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvaluationFormValues,
  type DatasetBindings,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import {
  lastExchange,
  useDatasetPreview,
} from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * The Jinja expressions templates should use for input / reference / context.
 *
 * Two shapes, decided by the dataset rather than by the user:
 *
 * - **OpenAI messages format.** The whole array binds to canonical ``messages``
 *   (``field_mapping`` refuses a path containing ``[`` or ``]``, so the index
 *   cannot live in the binding) and templates index it positionally. The user
 *   turn is the input and the assistant turn is the ground truth; there is no
 *   ambiguity to ask about.
 * - **Flat columns.** Canonical ``{{input}}`` / ``{{reference}}`` / ``{{context}}``,
 *   backed by the bindings the user picked.
 */
export function useDatasetBindings(): DatasetBindings {
  const { control } = useFormContext<EvaluationFormValues>();
  const [dataset, fieldMapping] = useWatch({ control, name: ['dataset', 'fieldMapping'] });
  const { messagesColumn, messageSelectors } = useDatasetPreview(dataset ?? null);

  return useMemo(() => {
    if (messagesColumn) {
      const { user, assistant } = lastExchange(messageSelectors);
      const asCanonical = (selector: string | null) =>
        selector ? `{{ ${selector.replace(messagesColumn, 'messages')} }}` : null;
      return {
        messagesColumn,
        // No `{{input}}` fallback: a messages dataset binds no canonical input
        // column, so that expression would fail the job at render time.
        input: asCanonical(user) ?? '',
        reference: asCanonical(assistant),
        context: null,
        inputPath: user,
        referencePath: assistant,
      };
    }

    return {
      messagesColumn: null,
      input: '{{input}}',
      reference: fieldMapping?.reference ? '{{reference}}' : null,
      context: fieldMapping?.context ? '{{context}}' : null,
      inputPath: fieldMapping?.input || null,
      referencePath: fieldMapping?.reference || null,
    };
  }, [messagesColumn, messageSelectors, fieldMapping]);
}
