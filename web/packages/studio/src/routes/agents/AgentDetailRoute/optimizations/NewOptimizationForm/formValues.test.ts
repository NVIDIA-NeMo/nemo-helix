// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_NAME_MAX_LENGTH } from '@studio/components/evaluation/submitEvaluationJob';
import { optimizationFormSchema } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { intentById } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';

const answers = (name: string) => ({
  name,
  intent: 'accuracy',
  budget: 'quick',
  experimentId: 'exp-1',
  judgeModel: 'default/judge',
  searchSpace: intentById('accuracy').parameters,
});

describe('optimizationFormSchema name', () => {
  it('accepts a name as long as the study job allows', () => {
    expect(optimizationFormSchema.safeParse(answers('a'.repeat(JOB_NAME_MAX_LENGTH))).success).toBe(
      true
    );
  });

  it('rejects a name the job service would refuse, though it is a valid entity name', () => {
    const result = optimizationFormSchema.safeParse(answers('a'.repeat(JOB_NAME_MAX_LENGTH + 1)));

    expect(result.success).toBe(false);
    expect(result.error?.issues[0].message).toBe(
      `Name must be ${JOB_NAME_MAX_LENGTH} characters or fewer.`
    );
  });
});
