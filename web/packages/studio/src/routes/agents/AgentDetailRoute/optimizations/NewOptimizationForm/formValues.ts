// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sanitizeEntityName, toValidEntityName } from '@nemo/common/src/utils/entityName';
import {
  type BudgetId,
  type IntentId,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizeConfig';
import { z } from 'zod';

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

/** A bound as the form holds it. Blank input becomes ``undefined`` rather than coercing to 0,
 *  which would silently pass validation as a real bound. */
const boundSchema = z.preprocess(
  (value) => (typeof value === 'string' && value.trim() === '' ? undefined : value),
  z.coerce.number({
    required_error: 'Enter a number.',
    invalid_type_error: 'Enter a number.',
  })
);

const searchParameterSchema = z
  .object({
    path: z.string().min(1),
    label: z.string().min(1),
    type: z.enum(['int', 'float']),
    low: boundSchema,
    high: boundSchema,
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
  experimentId: z.string().min(1, 'Pick an evaluation to score trials against.'),
  judgeModel: z.string().min(1, 'Pick a judge model to score trials with.'),
  searchSpace: z.array(searchParameterSchema).min(1, 'Sweep at least one parameter.'),
});

export type OptimizationFormValues = {
  name: string;
  intent: IntentId;
  budget: BudgetId;
  experimentId: string;
  judgeModel: string;
  searchSpace: SearchParameter[];
};

export type OptimizationFormOutput = z.output<typeof optimizationFormSchema>;
