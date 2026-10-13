// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parquetBatchGroups } from '@studio/components/evaluation/shared/parquetBatchGroups';

const DIR = 'results/a1/artifacts/dataset/parquet-files/';

describe('parquetBatchGroups', () => {
  it('groups the Parquet files of a folder under one glob', () => {
    expect(
      parquetBatchGroups([
        `${DIR}batch_00001.parquet`,
        `${DIR}batch_00000.parquet`,
        `${DIR}batch_00002.parquet`,
      ])
    ).toEqual([
      { glob: `${DIR}*.parquet`, dir: DIR, count: 3, firstPath: `${DIR}batch_00000.parquet` },
    ]);
  });

  it('keeps sibling and nested folders apart', () => {
    expect(
      parquetBatchGroups([
        `${DIR}batch_00000.parquet`,
        `${DIR}nested/batch_00001.parquet`,
        'results/a1/artifacts/dataset/tmp-partial-parquet-files/batch_00001.parquet',
      ])
    ).toEqual([]);
  });

  it('groups at the fileset root', () => {
    expect(parquetBatchGroups(['b.parquet', 'a.parquet', 'rows.jsonl'])).toEqual([
      { glob: '*.parquet', dir: '', count: 2, firstPath: 'a.parquet' },
    ]);
  });

  it('returns one group per folder, in folder order', () => {
    expect(
      parquetBatchGroups(['z/1.parquet', 'z/2.parquet', 'a/1.parquet', 'a/2.parquet']).map(
        (group) => group.glob
      )
    ).toEqual(['a/*.parquet', 'z/*.parquet']);
  });

  it("skips a job fileset's log shards", () => {
    const logs = 'logs/job=dd/job_attempt=a1/job_step=data-designer-job/job_task=t1/';
    expect(
      parquetBatchGroups([
        `${logs}0cc2.parquet`,
        `${logs}3d2a.parquet`,
        `${DIR}batch_00000.parquet`,
        `${DIR}batch_00001.parquet`,
      ]).map((group) => group.glob)
    ).toEqual([`${DIR}*.parquet`]);
  });

  it('skips a folder whose name would read as a glob', () => {
    expect(parquetBatchGroups(['run[1]/a.parquet', 'run[1]/b.parquet'])).toEqual([]);
  });
});
