// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parquetBatchGlob } from '@studio/components/evaluation/shared/parquetBatchGlob';

const DIR = 'results/a1/artifacts/dataset/parquet-files/';

describe('parquetBatchGlob', () => {
  it('globs every Parquet file beside the picked one', () => {
    expect(
      parquetBatchGlob(`${DIR}batch_00000.parquet`, [
        `${DIR}batch_00000.parquet`,
        `${DIR}batch_00001.parquet`,
        `${DIR}batch_00002.parquet`,
      ])
    ).toEqual({ glob: `${DIR}*.parquet`, dir: DIR, count: 3 });
  });

  it('ignores Parquet in sibling and nested directories', () => {
    expect(
      parquetBatchGlob(`${DIR}batch_00000.parquet`, [
        `${DIR}batch_00000.parquet`,
        `${DIR}nested/batch_00001.parquet`,
        'results/a1/artifacts/dataset/tmp-partial-parquet-files/batch_00001.parquet',
      ])
    ).toBeNull();
  });

  it('globs at the fileset root', () => {
    expect(parquetBatchGlob('a.parquet', ['a.parquet', 'b.parquet', 'rows.jsonl'])).toEqual({
      glob: '*.parquet',
      dir: '',
      count: 2,
    });
  });

  it('offers nothing for a file that is not Parquet', () => {
    expect(parquetBatchGlob('a.jsonl', ['a.jsonl', 'b.parquet', 'c.parquet'])).toBeNull();
  });

  it('offers nothing when the directory name would read as a glob', () => {
    expect(
      parquetBatchGlob('run[1]/a.parquet', ['run[1]/a.parquet', 'run[1]/b.parquet'])
    ).toBeNull();
  });
});
