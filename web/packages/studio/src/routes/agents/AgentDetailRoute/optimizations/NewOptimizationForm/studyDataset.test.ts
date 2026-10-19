// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readParquetRows } from '@studio/api/datasets/filesetParquetRows';
import type {
  DatasetEvalSpec,
  PersistedEvalSpec,
} from '@studio/components/evaluation/submitEvaluationJob';
import {
  type DatasetSource,
  globToRegExp,
  parseDatasetRecords,
  readDatasetRecords,
  studyEvaluationFromSpec,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/studyDataset';
import { gzipSync } from 'node:zlib';

vi.mock('@studio/api/datasets/filesetParquetRows', () => ({ readParquetRows: vi.fn() }));

const textBlob = (text: string) => new Blob([text]);
const unexpected = () => Promise.reject(new Error('unexpected dataset read'));
const noDataset: DatasetSource = { listFiles: unexpected, download: unexpected };
const sourceOf = (files: Record<string, string>): DatasetSource => ({
  listFiles: vi.fn(async () => Object.keys(files)),
  download: vi.fn(async (_workspace: string, _fileset: string, path: string) =>
    textBlob(files[path])
  ),
});

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
    const source = sourceOf({ 'dataset.jsonl': JSON.stringify(rows[0]) });
    await expect(studyEvaluationFromSpec(spec, source)).resolves.toEqual({
      spec,
      records: rows,
    });
    expect(source.download).toHaveBeenCalledWith('ws', 'data', 'dataset.jsonl');
    expect(source.listFiles).not.toHaveBeenCalled();
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
        sourceOf({ 'test.csv': 'subject,label,tool\n"Quota, urgent",phishing,triage\n' })
      )
    ).resolves.toMatchObject({
      records: [{ subject: 'Quota, urgent', label: 'phishing', tool: 'triage' }],
    });
  });

  it('concatenates every file a glob ref matches, in sorted path order', async () => {
    const spec: DatasetEvalSpec = { dataset: 'ws/data#dataset/*.jsonl', metrics: [] };
    const source = sourceOf({
      'dataset/part-1.jsonl': '{"n": 2}',
      'dataset/part-0.jsonl': '{"n": 1}',
      'dataset/nested/part-2.jsonl': '{"n": 3}',
      'other.jsonl': '{"n": 4}',
    });
    await expect(studyEvaluationFromSpec(spec, source)).resolves.toMatchObject({
      records: [{ n: 1 }, { n: 2 }],
    });
    expect(source.listFiles).toHaveBeenCalledWith('ws', 'data', 'dataset/');
  });

  it('names a glob ref that matches no files', async () => {
    await expect(
      studyEvaluationFromSpec(
        { dataset: 'ws/data#dataset/*.parquet', metrics: [] },
        sourceOf({ 'dataset.jsonl': '{}' })
      )
    ).rejects.toThrow(
      /No files in fileset "data" match the evaluation's dataset dataset\/\*\.parquet/
    );
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
    expect(() => parseDatasetRecords('ARROW1', 'rows.feather')).toThrow(
      /rows\.feather is not JSON/
    );
  });
});

describe('globToRegExp', () => {
  it.each([
    ['dataset/*.parquet', 'dataset/part-0.parquet', true],
    ['dataset/*.parquet', 'dataset/a/part-0.parquet', false],
    ['dataset/**/*.parquet', 'dataset/part-0.parquet', true],
    ['dataset/**/*.parquet', 'dataset/a/b/part-0.parquet', true],
    ['part-?.json', 'part-1.json', true],
    ['part-?.json', 'part-10.json', false],
    ['part-[01].json', 'part-1.json', true],
    ['part-[!01].json', 'part-1.json', false],
    ['data.v1/*.json', 'dataXv1/a.json', false],
  ])('%s against %s is %s', (pattern, path, expected) => {
    expect(globToRegExp(pattern).test(path)).toBe(expected);
  });
});

describe('readDatasetRecords', () => {
  it('decodes every row of a Parquet dataset', async () => {
    const rows = [{ q: 'hi' }];
    vi.mocked(readParquetRows).mockResolvedValue(rows);
    const blob = textBlob('PAR1');
    await expect(readDatasetRecords(blob, 'dataset.parquet')).resolves.toEqual(rows);
    expect(readParquetRows).toHaveBeenCalledWith(blob);
  });

  it('decompresses gzip before parsing the inner format', async () => {
    const blob = new Blob([gzipSync('{"a": 1}\n{"a": 2}\n')]);
    await expect(readDatasetRecords(blob, 'rows.jsonl.gz')).resolves.toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('names a file that claims gzip but is not', async () => {
    await expect(readDatasetRecords(textBlob('plain'), 'rows.csv.gz')).rejects.toThrow(
      /Could not decompress rows\.csv\.gz as gzip/
    );
  });
});
