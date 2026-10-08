// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DatasetEvalSpec,
  PersistedEvalSpec,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  parseDatasetRecords,
  studyEvaluationFromSpec,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';

const noDataset = () => Promise.reject(new Error('unexpected dataset read'));

describe('studyEvaluationFromSpec', () => {
  it('keeps task inputs, references, metrics and views intact without inventing an answer', async () => {
    const spec: PersistedEvalSpec = {
      tasks: [
        {
          id: 't1',
          intent: 'Say hi',
          inputs: { instruction: 'Hi' },
          reference: {},
          metrics: [],
          views: {},
        },
      ],
    };
    await expect(studyEvaluationFromSpec(spec, noDataset)).resolves.toEqual({
      spec,
      records: spec.tasks,
    });
  });

  it('retains all row columns without requiring a canonical reference', async () => {
    const spec: DatasetEvalSpec = {
      dataset: [
        {
          user_message: 'legit?',
          emails: [{ subject: 'Quota' }],
          expected_answer: 'phishing',
          expected_tool: 'triage_message',
        },
      ],
      prompt_template: '{{ item.user_message | upper }}',
      metrics: [],
    };
    await expect(studyEvaluationFromSpec(spec, noDataset)).resolves.toEqual({
      spec,
      records: spec.dataset,
    });
  });

  it('downloads the selected evaluation dataset and preserves its grading configuration', async () => {
    const spec: DatasetEvalSpec = {
      dataset: 'ws/data#dataset.jsonl',
      prompt_template: '{{ item.q }}',
      field_mapping: { reference: 'gold' },
      metrics: [],
    };
    const rows = [{ q: 'hi', gold: 'hello' }];
    const readDataset = vi.fn().mockResolvedValue(JSON.stringify(rows[0]));
    await expect(studyEvaluationFromSpec(spec, readDataset)).resolves.toEqual({
      spec,
      records: rows,
    });
    expect(readDataset).toHaveBeenCalledWith('ws', 'data', 'dataset.jsonl');
  });

  it('reads a CSV dataset without discarding its answer and routing columns', async () => {
    const spec: DatasetEvalSpec = {
      dataset: 'ws/data#test.csv',
      prompt_template: '{{ item.subject }}',
      metrics: [],
    };
    await expect(
      studyEvaluationFromSpec(
        spec,
        vi.fn().mockResolvedValue('subject,label,tool\n"Quota, urgent",phishing,triage\n')
      )
    ).resolves.toMatchObject({
      records: [{ subject: 'Quota, urgent', label: 'phishing', tool: 'triage' }],
    });
  });

  it('rejects a dataset that is not a fileset ref', async () => {
    await expect(
      studyEvaluationFromSpec({ dataset: 'hf://some/dataset', metrics: [] }, noDataset)
    ).rejects.toThrow(/not a workspace\/fileset/);
  });
});

describe('parseDatasetRecords', () => {
  it('reads JSON arrays and JSON lines', () => {
    expect(parseDatasetRecords('[{"a": 1}]')).toEqual([{ a: 1 }]);
    expect(parseDatasetRecords('{"a": 1}\n{"a": 2}\n')).toEqual([{ a: 1 }, { a: 2 }]);
  });
  it('rejects records that are not objects', () => {
    expect(() => parseDatasetRecords('[1, 2]')).toThrow(/JSON object/);
  });
  it('names a binary dataset format it cannot parse', () => {
    expect(() => parseDatasetRecords('PAR1', 'rows.parquet')).toThrow(/rows\.parquet is not JSON/);
  });
});
