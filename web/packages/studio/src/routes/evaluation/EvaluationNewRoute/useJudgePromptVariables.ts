// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { VariableDef } from '@nemo/common/src/components/form/VariableTextArea';
import {
  classifyJudgePromptVariables,
  isKnownJudgeVariable,
  type JudgePromptVariableReport,
} from '@studio/routes/evaluation/EvaluationNewRoute/judgePromptVariables';
import {
  CANONICAL_FIELD_LABELS,
  composeJudgeUserPrompt,
  type EvaluationFormValues,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { useDatasetBindings } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetBindings';
import { useDatasetPreview } from '@studio/routes/evaluation/EvaluationNewRoute/useDatasetPreview';
import { useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';

/** Strips the braces a binding already carries; `insertVariable` puts them back. */
const bare = (expression: string) => expression.replace(/^\{\{\s*|\s*\}\}$/g, '');

export interface JudgePromptVariables extends JudgePromptVariableReport {
  /** Offered for insertion and completion. Canonical names only. */
  available: VariableDef[];
  isKnown: (token: string) => boolean;
}

export function useJudgePromptVariables(): JudgePromptVariables {
  const { control } = useFormContext<EvaluationFormValues>();
  const [dataset, fieldMapping, judgePrompt] = useWatch({
    control,
    name: ['dataset', 'fieldMapping', 'body.judgePrompt'],
  });
  const bindings = useDatasetBindings();
  const { keyOptions } = useDatasetPreview(dataset ?? null);

  const columns = useMemo(() => keyOptions.map((option) => option.value), [keyOptions]);
  const mapping = useMemo(() => fieldMapping ?? {}, [fieldMapping]);
  const prompt = judgePrompt ?? composeJudgeUserPrompt(bindings);

  const available = useMemo<VariableDef[]>(() => {
    const mappedTo = (field: 'input' | 'reference' | 'context') =>
      mapping[field] ? ` (mapped to \`${mapping[field]}\`)` : '';

    const defs: VariableDef[] = [];
    if (bindings.input) {
      defs.push({
        name: bare(bindings.input),
        description: bindings.messagesColumn
          ? 'The user turn sent to the model'
          : `The ${CANONICAL_FIELD_LABELS.input} field${mappedTo('input')}`,
      });
    }
    if (bindings.reference) {
      defs.push({
        name: bare(bindings.reference),
        description: bindings.messagesColumn
          ? 'The assistant turn, as the reference'
          : `The ${CANONICAL_FIELD_LABELS.reference} field${mappedTo('reference')}`,
      });
    }
    if (bindings.context) {
      defs.push({
        name: bare(bindings.context),
        description: `The ${CANONICAL_FIELD_LABELS.context} field${mappedTo('context')}`,
      });
    }
    defs.push({
      name: 'sample.output_text',
      description: "The model's response, which is what gets graded",
    });
    return defs;
  }, [bindings, mapping]);

  const report = useMemo(
    () => classifyJudgePromptVariables(prompt, columns, mapping),
    [prompt, columns, mapping]
  );

  const isKnown = useMemo(() => isKnownJudgeVariable(columns, mapping), [columns, mapping]);

  return { ...report, available, isKnown };
}
