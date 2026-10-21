// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_NAME_MAX_LENGTH } from '@studio/components/evaluation/submitEvaluationJob';
import {
  DEFAULT_PROMPT_OPTIONS,
  optimizationFormSchema,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/formValues';
import { intentById } from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';

const answers = (name: string, prompt = DEFAULT_PROMPT_OPTIONS) => ({
  name,
  intent: 'accuracy',
  budget: 'quick',
  experimentId: 'exp-1',
  judgeModel: 'default/judge',
  searchSpace: intentById('accuracy').parameters,
  prompt,
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

describe('optimizationFormSchema prompt', () => {
  it('ignores the prompt fields while the phase is off', () => {
    expect(optimizationFormSchema.safeParse(answers('study')).success).toBe(true);
  });

  it('requires a purpose and a model once the phase is on', () => {
    const result = optimizationFormSchema.safeParse(
      answers('study', { ...DEFAULT_PROMPT_OPTIONS, enabled: true, purpose: '   ' })
    );

    expect(result.success).toBe(false);
    expect(result.error?.issues.map((issue) => [issue.path.join('.'), issue.message])).toEqual([
      ['prompt.purpose', 'Describe what the prompt should achieve.'],
      ['prompt.model', 'Pick a model to rewrite the prompt.'],
    ]);
  });

  it('trims the purpose it submits', () => {
    const result = optimizationFormSchema.parse(
      answers('study', {
        ...DEFAULT_PROMPT_OPTIONS,
        enabled: true,
        purpose: '  Answer accurately.  ',
        model: 'ws/rewriter',
      })
    );

    expect(result.prompt.purpose).toBe('Answer accurately.');
  });
});

describe('optimizationFormSchema prompt versions', () => {
  const versions = (list: string[]) =>
    answers('study', {
      ...DEFAULT_PROMPT_OPTIONS,
      enabled: true,
      mode: 'versions',
      versions: list,
    });

  it('needs no rewriter or purpose', () => {
    expect(optimizationFormSchema.safeParse(versions(['A', 'B'])).success).toBe(true);
  });

  it('submits only distinct, non-blank versions', () => {
    expect(
      optimizationFormSchema.parse(versions(['A', '  ', 'B', 'A ', '\nA'])).prompt.versions
    ).toEqual(['A', 'B']);
  });

  it('requires two different versions to sweep', () => {
    const result = optimizationFormSchema.safeParse(versions(['A', 'A', '']));

    expect(result.success).toBe(false);
    expect(result.error?.issues[0].message).toBe('Add at least two different prompt versions.');
  });
});
