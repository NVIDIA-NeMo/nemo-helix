// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import { JOB_NAME_MAX_LENGTH } from '@studio/components/evaluation/submitEvaluationJob';
import {
  type BudgetId,
  type IntentId,
  type PromptModeId,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { z } from 'zod';

/** Entity-naming contract: the literal typed value is never rewritten, and the sanitized name is
 *  what reaches submit — so a cosmetic deviation is a preview concern, not a validation error.
 *  Only input `sanitizeEntityName` cannot salvage becomes an issue, or a name too long for the
 *  study's job, whose output fileset is named after it. */
export const nameSchema = z
  .string()
  .superRefine((value, ctx) => {
    const sanitized = sanitizeEntityName(value);
    if (sanitized === undefined) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: value
          ? 'Name must start with a letter and be at least 2 characters.'
          : 'Name is required.',
      });
    } else if (sanitized.length > JOB_NAME_MAX_LENGTH) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `Name must be ${JOB_NAME_MAX_LENGTH} characters or fewer.`,
      });
    }
  })
  .transform((value) => toValidEntityName(value, value));

const searchParameterSchema = z
  .object({
    path: z.string().min(1),
    label: z.string().min(1),
    type: z.enum(['int', 'float']),
    low: z.coerce.number({ invalid_type_error: 'Enter a number.' }),
    high: z.coerce.number({ invalid_type_error: 'Enter a number.' }),
  })
  .superRefine((parameter, ctx) => {
    if (parameter.high <= parameter.low) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Max must be greater than min.',
        path: ['high'],
      });
    }
  });

/** Versions worth sweeping: trimmed, blanks dropped, duplicates collapsed so a trial is never spent
 *  re-running a prompt another trial already ran. */
export const sweepableVersions = (versions: string[]): string[] => [
  ...new Set(versions.map((version) => version.trim()).filter(Boolean)),
];

/** What keeps the prompt step from running, in page order. Its fields are only required once the
 *  step is switched on, and then only those of the chosen mode, so a user who never opens it is
 *  not blocked by them. Shared by the schema and the run panel so the two cannot disagree. */
export const promptIssues = (
  prompt: PromptOptionsValues
): { path: keyof PromptOptionsValues; message: string }[] => {
  if (!prompt.enabled) return [];
  if (prompt.mode === 'versions') {
    return sweepableVersions(prompt.versions).length < 2
      ? [{ path: 'versions', message: 'Add at least two different prompt versions.' }]
      : [];
  }
  return [
    ...(prompt.purpose.trim()
      ? []
      : [{ path: 'purpose' as const, message: 'Describe what the prompt should achieve.' }]),
    ...(prompt.model
      ? []
      : [{ path: 'model' as const, message: 'Pick a model to rewrite the prompt.' }]),
  ];
};

const promptOptionsSchema = z
  .object({
    enabled: z.boolean(),
    mode: z.enum(['rewrite', 'versions']),
    purpose: z.string().trim(),
    model: z.string(),
    versions: z.array(z.string()).transform(sweepableVersions),
  })
  .superRefine((prompt, ctx) =>
    promptIssues(prompt).forEach(({ path, message }) =>
      ctx.addIssue({ code: z.ZodIssueCode.custom, message, path: [path] })
    )
  );

export const optimizationFormSchema = z.object({
  name: nameSchema,
  intent: z.enum(['accuracy', 'brevity', 'creativity', 'cost']),
  budget: z.enum(['quick', 'standard', 'thorough']),
  experimentId: z.string().min(1, 'Pick an evaluation to score trials against.'),
  searchSpace: z.array(searchParameterSchema).min(1, 'Sweep at least one parameter.'),
  prompt: promptOptionsSchema,
});

export interface PromptOptionsValues {
  enabled: boolean;
  mode: PromptModeId;
  /** `rewrite` only: what every rewritten version must stay true to. */
  purpose: string;
  /** `rewrite` only: model URN the GA calls through the gateway to mutate and recombine prompts. */
  model: string;
  /** `versions` only: the user's prompts, swept as one categorical dimension. */
  versions: string[];
}

export const DEFAULT_PROMPT_OPTIONS: PromptOptionsValues = {
  enabled: false,
  mode: 'rewrite',
  purpose: '',
  model: '',
  versions: ['', ''],
};

/** What the fields hold while editing — `name` before its transform, numbers still possibly
 *  mid-keystroke — as opposed to `z.output`, which is the submitted shape. */
export type OptimizationFormValues = {
  name: string;
  intent: IntentId;
  budget: BudgetId;
  experimentId: string;
  searchSpace: SearchParameter[];
  prompt: PromptOptionsValues;
};

export type OptimizationFormOutput = z.output<typeof optimizationFormSchema>;
