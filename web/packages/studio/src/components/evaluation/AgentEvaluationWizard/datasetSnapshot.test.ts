// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getFilesListFilesetFilesQueryKey } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse } from '@nemo/sdk/generated/platform/schema';
import {
  DATASET_SNAPSHOT_KEY,
  datasetDrift,
  datasetSnapshotOf,
  takeDatasetSnapshot,
} from '@studio/components/evaluation/AgentEvaluationWizard/datasetSnapshot';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { http, HttpResponse } from 'msw';

const listFiles = (files: { path: string; size: number }[] | 'gone' | 'error') =>
  server.use(
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () => {
      if (files === 'gone') return new HttpResponse(null, { status: 404 });
      if (files === 'error') return new HttpResponse(null, { status: 500 });
      return HttpResponse.json({
        data: files.map(({ path, size }) => ({ path, size, file_ref: path, file_url: path })),
      });
    })
  );

const FILE_REF = 'ws/data#rows/train.jsonl';
const GLOB_REF = 'ws/data#output/*.parquet';

describe('takeDatasetSnapshot', () => {
  it('records only the file a ref names, not its siblings', async () => {
    listFiles([
      { path: 'rows/train.jsonl', size: 10 },
      { path: 'rows/train.jsonl.bak', size: 9 },
      { path: 'rows/test.jsonl', size: 8 },
    ]);

    expect(await takeDatasetSnapshot('ws', FILE_REF)).toEqual({
      ref: FILE_REF,
      files: [{ path: 'rows/train.jsonl', size: 10 }],
    });
  });

  it('records every file a glob matches in its own folder', async () => {
    listFiles([
      { path: 'output/part-0.parquet', size: 10 },
      { path: 'output/part-1.parquet', size: 11 },
      { path: 'output/nested/part-2.parquet', size: 12 },
      { path: 'output/part-3.parquetx', size: 13 },
    ]);

    expect(await takeDatasetSnapshot('ws', GLOB_REF)).toEqual({
      ref: GLOB_REF,
      files: [
        { path: 'output/part-0.parquet', size: 10 },
        { path: 'output/part-1.parquet', size: 11 },
      ],
    });
  });

  it('records nothing when no file matches or the fileset is gone', async () => {
    listFiles([{ path: 'other.jsonl', size: 1 }]);
    expect(await takeDatasetSnapshot('ws', FILE_REF)).toBeNull();

    listFiles('gone');
    expect(await takeDatasetSnapshot('ws', FILE_REF)).toBeNull();
  });
});

describe('datasetSnapshotOf', () => {
  const withMetadata = (value?: string): EvaluationResponse =>
    ({
      id: 'eval_1',
      name: 'run',
      workspace: 'ws',
      experiment_ids: [],
      metadata: value === undefined ? {} : { [DATASET_SNAPSHOT_KEY]: value },
    }) as unknown as EvaluationResponse;

  it('reads a recorded snapshot', () => {
    const snapshot = { ref: FILE_REF, files: [{ path: 'rows/train.jsonl', size: 10 }] };
    expect(datasetSnapshotOf(withMetadata(JSON.stringify(snapshot)))).toEqual(snapshot);
  });

  it.each([
    ['no snapshot', undefined],
    ['malformed JSON', '{not json'],
    ['the wrong shape', JSON.stringify({ ref: FILE_REF, files: [{ path: 1 }] })],
  ])('treats %s as no snapshot', (_, value) => {
    expect(datasetSnapshotOf(withMetadata(value))).toBeNull();
  });
});

describe('datasetDrift', () => {
  const snapshot = {
    ref: GLOB_REF,
    files: [
      { path: 'output/part-0.parquet', size: 10 },
      { path: 'output/part-1.parquet', size: 11 },
    ],
  };

  it('reports nothing when every file is still there at the same size', async () => {
    listFiles(snapshot.files);
    expect(await datasetDrift('ws', snapshot)).toBeNull();
  });

  it.each([
    ['a size changed', [snapshot.files[0], { path: 'output/part-1.parquet', size: 12 }]],
    ['a file was added', [...snapshot.files, { path: 'output/part-2.parquet', size: 1 }]],
    ['a file was removed', [snapshot.files[0]]],
  ])('reports a change when %s', async (_, files) => {
    listFiles(files);
    expect(await datasetDrift('ws', snapshot)).toBe('changed');
  });

  it('reports the dataset missing when the fileset is gone or no file matches', async () => {
    listFiles('gone');
    expect(await datasetDrift('ws', snapshot)).toBe('missing');

    listFiles([{ path: 'elsewhere/part-0.parquet', size: 10 }]);
    expect(await datasetDrift('ws', snapshot)).toBe('missing');
  });

  it('passes on any other listing failure', async () => {
    listFiles('error');
    await expect(datasetDrift('ws', snapshot)).rejects.toBeDefined();
  });
});
