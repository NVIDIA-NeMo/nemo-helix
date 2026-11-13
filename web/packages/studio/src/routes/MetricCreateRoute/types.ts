// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  entityNameField,
  unsalvageableNameError,
} from '@studio/components/evaluation/EvalWizard/entityNameField';
import {
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
  type MetricConfig,
  NUMBER_CHECK_OPERATIONS,
  type SelectableMetric,
  STRING_CHECK_OPERATIONS,
} from '@studio/components/evaluation/EvalWizard/types';
import type { PanelScoreFormData } from '@studio/components/evaluation/Jobs/form/ScoreModal';
import { z } from 'zod';

export interface MetricFormValues {
  name: string;
  description: string;
  type: SelectableMetric;
  body: MetricConfig;
}

export interface MetricTypeOption {
  type: SelectableMetric;
  label: string;
  description: string;
}

export interface MetricTypeGroup {
  heading: string;
  options: MetricTypeOption[];
}

/** Grouped by what the user is trying to do, not by algorithm name. */
export const METRIC_TYPE_GROUPS: MetricTypeGroup[] = [
  {
    heading: 'Judge with a model',
    options: [
      {
        type: 'llm-judge',
        label: 'LLM-as-a-Judge',
        description: 'A model grades each response against criteria you define.',
      },
    ],
  },
  {
    heading: 'Compare to a reference',
    options: [
      {
        type: 'exact-match',
        label: 'Exact Match',
        description: 'Scores 1 when the response equals the reference, otherwise 0.',
      },
      {
        type: 'f1',
        label: 'F1',
        description: 'Token overlap between the response and the reference.',
      },
      {
        type: 'rouge',
        label: 'ROUGE',
        description: 'N-gram recall against the reference. Suits summaries.',
      },
      {
        type: 'bleu',
        label: 'BLEU',
        description: 'N-gram precision against the reference. Suits translation.',
      },
    ],
  },
  {
    heading: 'Check a rule',
    options: [
      {
        type: 'string-check',
        label: 'String Check',
        description: 'Passes when the response contains, starts with, or equals the reference.',
      },
      {
        type: 'number-check',
        label: 'Number Check',
        description: 'Compares the number in the response to the reference.',
      },
    ],
  },
];

/** A stored metric is dataset-agnostic, so templates name the canonical fields;
 *  each evaluation's field mapping decides where they come from. */
export const CANONICAL_BINDINGS: DatasetBindings = {
  messagesColumn: null,
  input: '{{input}}',
  reference: '{{reference}}',
  context: null,
  inputPath: null,
  referencePath: null,
};

export const METRIC_FORM_DEFAULTS: MetricFormValues = {
  name: '',
  description: '',
  type: 'llm-judge',
  body: {
    scores: EVALUATION_FORM_DEFAULTS.body.scores,
    judgeModel: '',
    judgePrompt: null,
    stringCheck: EVALUATION_FORM_DEFAULTS.body.stringCheck,
    numberCheck: EVALUATION_FORM_DEFAULTS.body.numberCheck,
  },
};

const issue = (ctx: z.RefinementCtx, path: (string | number)[], message: string) =>
  ctx.addIssue({ code: z.ZodIssueCode.custom, path, message });

export const metricSchema = z
  .object({
    name: entityNameField(),
    description: z.string(),
    type: z.custom<SelectableMetric>(),
    body: z.object({
      judgeModel: z.string(),
      judgePrompt: z.string().nullable(),
      scores: z.array(z.custom<PanelScoreFormData>()),
      stringCheck: z.object({ operation: z.enum(STRING_CHECK_OPERATIONS) }),
      numberCheck: z.object({
        operation: z.enum(NUMBER_CHECK_OPERATIONS),
        epsilon: z.number().nullable(),
      }),
    }),
  })
  .superRefine((values, ctx) => {
    const nameError = unsalvageableNameError(values.name, 'Name');
    if (nameError) issue(ctx, ['name'], nameError);

    if (values.type === 'llm-judge') {
      if (!values.body.judgeModel) issue(ctx, ['body', 'judgeModel'], 'Select a judge model.');
      if (!values.body.scores.length)
        issue(ctx, ['body', 'scores'], 'Add at least one score for the judge to extract.');
    }

    if (
      values.type === 'number-check' &&
      values.body.numberCheck.operation === 'absolute difference' &&
      values.body.numberCheck.epsilon === null
    ) {
      issue(ctx, ['body', 'numberCheck', 'epsilon'], 'Absolute difference requires a tolerance.');
    }
  });
