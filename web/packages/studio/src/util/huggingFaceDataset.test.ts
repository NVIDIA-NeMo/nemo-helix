// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesetRowsQueryOptions } from '@studio/api/datasets/filesetParquetRows';
import type { CustomizationTemplateDataset } from '@studio/constants/customizationTemplates';
import { fetchAndConvertDataset } from '@studio/util/huggingFaceDataset';
import { QueryClient } from '@tanstack/react-query';

// Reading and decoding rows is covered in filesetParquetRows.test.ts. This suite owns the
// conversion contract: which rows land in which partition, and when that is refused.
vi.mock('@studio/api/datasets/filesetParquetRows', () => ({
  filesetRowsQueryOptions: vi.fn(),
}));

const rowsOptions = vi.mocked(filesetRowsQueryOptions);

type RowsParams = Parameters<typeof filesetRowsQueryOptions>[0];

/** Serves `rows` (or a failure) in place of a real fileset read. */
const serveRows = (rows: Record<string, unknown>[] | Error) => {
  rowsOptions.mockImplementation(
    (params: RowsParams) =>
      ({
        queryKey: ['test-rows', params.filesetName, params.rowCount],
        queryFn: () => (rows instanceof Error ? Promise.reject(rows) : Promise.resolve(rows)),
      }) as never
  );
};

const repeat = (count: number, row: Record<string, unknown> = { text: 'x' }) =>
  Array.from({ length: count }, () => ({ ...row }));

const dataset = (
  overrides: Partial<CustomizationTemplateDataset> = {}
): CustomizationTemplateDataset => ({
  hfRepoId: 'owner/ds',
  sourceFilesetName: 'owner-ds-hf',
  filePattern: /train-.*\.parquet$/,
  trainingRowCount: 3,
  validationRowCount: 2,
  name: 'test-dataset',
  convertRow: (row) => row,
  ...overrides,
});

describe('fetchAndConvertDataset', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient();
  });

  it('splits converted rows into training and validation JSONL blobs', async () => {
    serveRows(repeat(5, { text: 'hello' }));

    const { training, validation } = await fetchAndConvertDataset(
      queryClient,
      'ws',
      dataset(),
      () => {}
    );

    const trainingLines = (await training.text()).split('\n');
    const validationLines = (await validation.text()).split('\n');
    expect(trainingLines).toHaveLength(3);
    expect(validationLines).toHaveLength(2);
    expect(JSON.parse(trainingLines[0])).toEqual({ text: 'hello' });
  });

  it('reads the source fileset for exactly the rows both partitions need', async () => {
    serveRows(repeat(5));

    await fetchAndConvertDataset(queryClient, 'ws', dataset(), () => {});

    expect(rowsOptions).toHaveBeenCalledWith(
      expect.objectContaining({
        workspace: 'ws',
        filesetName: 'owner-ds-hf',
        rowCount: 5,
      })
    );
  });

  it('walks the caller through each phase', async () => {
    serveRows(repeat(5));
    const onProgress = vi.fn();

    await fetchAndConvertDataset(queryClient, 'ws', dataset(), onProgress);

    expect(onProgress.mock.calls.map(([phase]) => phase)).toEqual(['locating', 'converting']);
  });

  it('forwards download progress from the fileset read', async () => {
    rowsOptions.mockImplementation(
      (params: RowsParams) =>
        ({
          queryKey: ['test-rows'],
          queryFn: () => {
            params.onDownloadProgress?.(512, 1024);
            return Promise.resolve(repeat(5));
          },
        }) as never
    );
    const onProgress = vi.fn();

    await fetchAndConvertDataset(queryClient, 'ws', dataset(), onProgress);

    expect(onProgress).toHaveBeenCalledWith('downloading', 512, 1024);
  });

  it('does not backfill a dropped training row from the validation partition', async () => {
    serveRows([{ text: 'a' }, { drop: true }, { text: 'c' }, { text: 'd' }, { text: 'e' }]);

    await expect(
      fetchAndConvertDataset(
        queryClient,
        'ws',
        dataset({
          trainingRowCount: 3,
          validationRowCount: 2,
          convertRow: (row) => (row.drop ? null : row),
        }),
        () => {}
      )
    ).rejects.toThrow(/Not enough valid training rows: needed 3, found 2/);
  });

  it('throws when no rows survive conversion', async () => {
    serveRows(repeat(5));

    await expect(
      fetchAndConvertDataset(queryClient, 'ws', dataset({ convertRow: () => null }), () => {})
    ).rejects.toThrow(/Not enough valid training rows/);
  });

  it('throws when valid training rows are short of the configured count', async () => {
    serveRows([{ text: 'x' }, { text: 'x' }, { drop: true }]);

    await expect(
      fetchAndConvertDataset(
        queryClient,
        'ws',
        dataset({
          trainingRowCount: 3,
          validationRowCount: 0,
          convertRow: (row) => (row.drop ? null : row),
        }),
        () => {}
      )
    ).rejects.toThrow(/Not enough valid training rows: needed 3, found 2/);
  });

  it('throws when validation is requested but yields no rows', async () => {
    serveRows([{ text: 'x' }, { text: 'x' }, { text: 'x' }, { drop: true }, { drop: true }]);

    await expect(
      fetchAndConvertDataset(
        queryClient,
        'ws',
        dataset({ convertRow: (row) => (row.drop ? null : row) }),
        () => {}
      )
    ).rejects.toThrow(/Not enough valid validation rows/);
  });

  /** Parquet INT64 columns decode as BigInt, which plain JSON.stringify refuses. */
  it('serializes BigInt row values instead of throwing on them', async () => {
    serveRows(repeat(5, { id: 9007199254740993n, text: 'x' }));

    const { training } = await fetchAndConvertDataset(queryClient, 'ws', dataset(), () => {});

    expect(JSON.parse((await training.text()).split('\n')[0])).toEqual({
      id: '9007199254740993',
      text: 'x',
    });
  });

  it('surfaces a failure from the fileset read', async () => {
    serveRows(new Error('No dataset file matched /train-.*/.'));

    await expect(fetchAndConvertDataset(queryClient, 'ws', dataset(), () => {})).rejects.toThrow(
      /No dataset file matched/
    );
  });
});
