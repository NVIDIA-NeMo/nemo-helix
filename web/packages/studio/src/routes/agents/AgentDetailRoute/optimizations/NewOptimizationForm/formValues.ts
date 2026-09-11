// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  type BudgetId,
  type IntentId,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfig';
import { z } from 'zod';

/** Entity-naming contract: the literal typed value is never rewritten, and the sanitized name is
 *  what reaches submit — so a cosmetic deviation is a preview concern, not a validation error.
 *  Only input `sanitizeEntityName` cannot salvage becomes an issue. */
const nameSchema = z
  .string()
  .superRefine((value, ctx) => {
    if (sanitizeEntityName(value) === undefined) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: value ? 'Name must contain at least one letter or number.' : 'Name is required.',
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

export const optimizationFormSchema = z.object({
  name: nameSchema,
  intent: z.enum(['accuracy', 'brevity', 'creativity', 'cost']),
  budget: z.enum(['quick', 'standard', 'thorough']),
  /** Empty when the agent has no evaluation to score against, which blocks submission. */
  experimentId: z.string().min(1, 'Pick an evaluation to score trials against.'),
  /** The study re-scores every trial with its own LLM judge, so one has to be named. */
  judgeModel: z.string().min(1, 'Pick a judge model to score trials with.'),
  searchSpace: z.array(searchParameterSchema).min(1, 'Sweep at least one parameter.'),
});

/** What the fields hold while editing — `name` before its transform, numbers still possibly
 *  mid-keystroke — as opposed to `z.output`, which is the submitted shape. */
export type OptimizationFormValues = {
  name: string;
  intent: IntentId;
  budget: BudgetId;
  experimentId: string;
  judgeModel: string;
  searchSpace: SearchParameter[];
};

export type OptimizationFormOutput = z.output<typeof optimizationFormSchema>;
