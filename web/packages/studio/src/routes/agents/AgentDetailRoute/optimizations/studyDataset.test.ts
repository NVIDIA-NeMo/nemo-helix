// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvalSpec } from '@studio/components/evaluation/submitEvaluationJob';
import {
  StudyDatasetError,
  studyRowsFromEvalSpec,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/studyDataset';

const task = (id: string, instruction: string, answer?: string) => ({
  id,
  intent: instruction,
  inputs: { instruction },
  ...(answer ? { reference: { answer } } : {}),
  metrics: [],
});

describe('studyRowsFromEvalSpec', () => {
  it('reads a task-shaped config down to prompt and expected answer', () => {
    const spec = { tasks: [task('a', 'Is this phishing?', 'phishing')] } as unknown as EvalSpec;

    expect(studyRowsFromEvalSpec(spec)).toEqual([
      { id: 'a', instruction: 'Is this phishing?', answer: 'phishing' },
    ]);
  });

  it('falls back to the task intent when it carries no explicit instruction', () => {
    const spec = {
      tasks: [{ id: 'b', intent: 'Summarize the thread', metrics: [] }],
    } as unknown as EvalSpec;

    expect(studyRowsFromEvalSpec(spec)[0].instruction).toBe('Summarize the thread');
  });

  it('leaves the answer empty rather than failing when a task has no reference', () => {
    const spec = { tasks: [task('c', 'Anything?')] } as unknown as EvalSpec;

    expect(studyRowsFromEvalSpec(spec)[0].answer).toBe('');
  });

  it('reads a dataset-shaped config, collapsing the row field shorthands', () => {
    const spec = {
      dataset: [{ id: 7, question: 'Capital of France?', expected_answer: 'Paris' }],
      metrics: [],
    } as unknown as EvalSpec;

    expect(studyRowsFromEvalSpec(spec)).toEqual([
      { id: '7', instruction: 'Capital of France?', answer: 'Paris' },
    ]);
  });

  it('rejects a dataset named by reference, whose rows Studio cannot stage', () => {
    const spec = { dataset: 'default/some-fileset', metrics: [] } as unknown as EvalSpec;

    expect(() => studyRowsFromEvalSpec(spec)).toThrow(StudyDatasetError);
  });

  it('rejects a row with nothing to prompt the agent with', () => {
    const spec = { dataset: [{ answer: 'Paris' }], metrics: [] } as unknown as EvalSpec;

    expect(() => studyRowsFromEvalSpec(spec)).toThrow(/no instruction/);
  });

  it('rejects a config with no tasks at all', () => {
    expect(() => studyRowsFromEvalSpec({ tasks: [] } as unknown as EvalSpec)).toThrow(
      StudyDatasetError
    );
  });
});
