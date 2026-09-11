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
