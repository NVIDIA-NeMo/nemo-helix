// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DatasetEvalSpec,
  PersistedEvalSpec,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  parseDatasetRecords,
  studyRowsFromSpec,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';

const noDataset = () => Promise.reject(new Error('unexpected dataset read'));

describe('studyRowsFromSpec', () => {
  it('reads task-driven rows from each task instruction and reference', async () => {
    const spec: PersistedEvalSpec = {
      tasks: [
        {
          id: 't1',
          intent: 'Classify',
          inputs: { instruction: 'Is this phishing?' },
          reference: { label: 'phishing' },
          metrics: [],
        },
        { id: 't2', intent: 'Say hi', reference: { a: 1, b: 2 }, metrics: [] },
      ],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).resolves.toEqual([
      { id: 't1', question: 'Is this phishing?', answer: 'phishing' },
      { id: 't2', question: 'Say hi', answer: '{"a":1,"b":2}' },
    ]);
  });

  it('renders simple prompt templates over inline dataset rows', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ id: 7, user_message: 'Capital of France?', reference: 'Paris' }],
      prompt_template: 'Q: {{ item.user_message }}',
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).resolves.toEqual([
      { id: '7', question: 'Q: Capital of France?', answer: 'Paris' },
    ]);
  });

  it('downloads a referenced dataset and renders its Jinja template per row', async () => {
    const spec: DatasetEvalSpec = {
      dataset: 'ws/data#dataset.jsonl',
      prompt_template:
        '{{ item.user_message }}{% for e in item.emails %}\n{{ loop.index }}. {{ e.subject }}{% endfor %}',
      field_mapping: { reference: 'expected_answer' },
      metrics: [],
    };
    const readDataset = vi
      .fn()
      .mockResolvedValue(
        '{"user_message": "legit?", "emails": [{"subject": "Quota"}], "expected_answer": "phishing"}\n'
      );

    await expect(studyRowsFromSpec(spec, readDataset)).resolves.toEqual([
      { id: '0', question: 'legit?\n1. Quota', answer: 'phishing' },
    ]);
    expect(readDataset).toHaveBeenCalledWith('ws', 'data', 'dataset.jsonl');
  });

  it('renders the last user message of a chat template, through the field mapping', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ q: 'Capital of France?', gold: 'Paris' }],
      prompt_template: {
        messages: [
          { role: 'system', content: 'Be brief.' },
          { role: 'user', content: '{{ input }}' },
        ],
      },
      field_mapping: { input: 'q', reference: 'gold' },
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).resolves.toEqual([
      { id: '0', question: 'Capital of France?', answer: 'Paris' },
    ]);
  });

  it('rejects a dataset evaluation with no prompt template', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ prompt: 'hi', reference: 'hello' }],
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(/has no prompt template/);
  });

  it('names the row and the construct when the template cannot be rendered', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ id: 1, q: 'hi' }],
      prompt_template: '{{ item.q | upper }}',
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(
      /dataset row 1: filter "upper" is not supported/
    );
  });

  it('rejects a task with no reference answer for the judge', async () => {
    const spec: PersistedEvalSpec = { tasks: [{ id: 't1', intent: 'Say hi', metrics: [] }] };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(
      /Task t1 has no reference answer/
    );
  });

  it('rejects a dataset row without the reference column, rather than guessing one', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ id: 3, prompt: 'hi', expected_answer: 'hello' }],
      prompt_template: '{{ item.prompt }}',
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(
      /Dataset row 3 has no expected answer .* from "reference"/
    );
  });

  it('names the mapped reference column when a row lacks it', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [{ id: 3, prompt: 'hi' }],
      prompt_template: '{{ item.prompt }}',
      field_mapping: { reference: 'gold' },
      metrics: [],
    };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(/from "gold"/);
  });

  it('reads a referenced CSV dataset, as the evaluator does', async () => {
    const spec: DatasetEvalSpec = {
      dataset: 'ws/data#smaller_test.csv',
      prompt_template: 'Subject: {{ item.subject }}',
      field_mapping: { reference: 'label' },
      metrics: [],
    };
    const readDataset = vi.fn().mockResolvedValue('subject,label\n"Quota, urgent",phishing\n');

    await expect(studyRowsFromSpec(spec, readDataset)).resolves.toEqual([
      { id: '0', question: 'Subject: Quota, urgent', answer: 'phishing' },
    ]);
  });

  it('rejects a dataset that is not a fileset ref', async () => {
    const spec: DatasetEvalSpec = { dataset: 'hf://some/dataset', metrics: [] };

    await expect(studyRowsFromSpec(spec, noDataset)).rejects.toThrow(/not a workspace\/fileset/);
  });
});

describe('parseDatasetRecords', () => {
  it('reads a JSON array and JSON lines alike', () => {
    expect(parseDatasetRecords('[{"a": 1}]')).toEqual([{ a: 1 }]);
    expect(parseDatasetRecords('{"a": 1}\n{"a": 2}\n')).toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('rejects records that are not objects', () => {
    expect(() => parseDatasetRecords('[1, 2]')).toThrow(/JSON object/);
  });

  it('names a binary dataset format it cannot read in the browser', () => {
    expect(() => parseDatasetRecords('PAR1', 'rows.parquet')).toThrow(/rows\.parquet is not JSON/);
  });
});
