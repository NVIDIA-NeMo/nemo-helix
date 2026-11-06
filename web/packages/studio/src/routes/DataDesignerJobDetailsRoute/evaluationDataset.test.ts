// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evaluationDatasetFromOutput } from '@studio/routes/DataDesignerJobDetailsRoute/evaluationDataset';

const FILESET = 'persona-tests-output';

describe('evaluationDatasetFromOutput', () => {
  it('reads every Parquet batch together', () => {
    expect(
      evaluationDatasetFromOutput(FILESET, [
        'builder_config.json',
        'parquet-files/batch_00001.parquet',
        'parquet-files/batch_00000.parquet',
        'logs/job.log',
      ])
    ).toEqual({
      fileset: FILESET,
      file: 'parquet-files/batch_00000.parquet',
      batchGlob: 'parquet-files/*.parquet',
    });
  });

  it('prefers the edits saved from the Data tab', () => {
    expect(
      evaluationDatasetFromOutput(FILESET, [
        'parquet-files/batch_00000.parquet',
        'parquet-files/batch_00000.edited.jsonl',
      ])
    ).toEqual({ fileset: FILESET, file: 'parquet-files/batch_00000.edited.jsonl' });
  });

  it('uses a lone data file as is', () => {
    expect(
      evaluationDatasetFromOutput(FILESET, ['builder_config.json', 'dataset.parquet'])
    ).toEqual({ fileset: FILESET, file: 'dataset.parquet' });
  });

  it('finds nothing when the output has no data files', () => {
    expect(evaluationDatasetFromOutput(FILESET, ['builder_config.json'])).toBeUndefined();
  });
});
