// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type EvaluationFormValues,
  type DatasetBindings,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/**
 * The Jinja expressions templates should use for input / reference / context:
 * always the canonical ``{{input}}`` / ``{{reference}}`` / ``{{context}}``, so a
 * saved configuration stays portable across datasets. Where each value lives in
 * the row is ``field_mapping``'s job -- a column for a flat file, a role path
 * into the array for an OpenAI messages file (see ``useMessagesBinding``).
 */
export function useDatasetBindings(): DatasetBindings {
  const { control } = useFormContext<EvaluationFormValues>();
  const fieldMapping = useWatch({ control, name: 'fieldMapping' });

  return useMemo(
    () => ({
      messagesColumn: fieldMapping?.messages || null,
      input: '{{input}}',
      reference: fieldMapping?.reference ? '{{reference}}' : null,
      context: fieldMapping?.context ? '{{context}}' : null,
      inputPath: fieldMapping?.input || null,
      referencePath: fieldMapping?.reference || null,
    }),
    [fieldMapping]
  );
}
