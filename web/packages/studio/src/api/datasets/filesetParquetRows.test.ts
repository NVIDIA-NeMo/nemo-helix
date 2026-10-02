// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { customFetch } from '@nemo/sdk/generated/fetchers/platform';
import { filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import type { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import {
  fetchFilesetRows,
  filesetRowsQueryOptions,
  matchDatasetFiles,
  readParquetRows,
  TransientFilesetError,
} from '@studio/api/datasets/filesetParquetRows';
import { AxiosError } from 'axios';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesListFilesetFiles: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/fetchers/platform', () => ({ customFetch: vi.fn() }));

const listMock = vi.mocked(filesListFilesetFiles);
const fetchMock = vi.mocked(customFetch);

/**
 * Real Parquet, decoded by the real `hyparquet` — only the bytes are canned, so the decode
 * under test still runs for real. Inline rather than generated at test time, to keep a
 * Parquet *writer* out of the dependency tree for three fixtures under a kilobyte total.
 *
 * Regenerate with hyparquet-writer (not a dependency; run it ad hoc):
 *   parquetWriteBuffer({ columnData: [{ name: 'n', data: ['a', 'b'] }] })
 * then base64-encode the returned buffer.
 */
const PARQUET = {
  /** `{ n: 'a' }, { n: 'b' }` */
  twoRows:
    'UEFSMRUGFRQVGFwVBBUAFQQVABUAFQAAAAokAQAAAGEBAAAAYhUEGSxIBHJvb3QVAgAVDCUAGAFuJQAAFgQZHBkcJggcFQwZFQAZGAFuFQIWBBZCFkImCDw2ACgBYhgBYQAZHBUGFQAVAgAAABZCFgQAKAloeXBhcnF1ZXQAWQAAAFBBUjE=',
  /** `{ n: 'row-0' } … { n: 'row-9' }` */
  tenRows:
    'UEFSMRUGFbQBFWhcFRQVABUUFQAVABUAAABaIAUAAAByb3ctMBEJADERCQAyEQkAMxEJADQRCQA1EQkANhEJADcRCSQ4BQAAAHJvdy05FQQZLEgEcm9vdBUCABUMJQAYAW4lAAAWFBkcGRwmCBwVDBkVABkYAW4VAhYUFpQBFpQBJgg8NgAoBXJvdy05GAVyb3ctMAAZHBUGFQAVAgAAABaUARYUACgJaHlwYXJxdWV0AGQAAABQQVIx',
  /** Two BIRD-SQL-shaped rows, the schema every shipped recipe's converter reads. */
  birdRows:
    'UEFSMRUGFRgVHFwVBBUAFQQVABUAFQAAAAwsAgAAAHExAgAAAHEyFQYVMBU0XBUEFQAVBBUAFQAVAAAAGFwIAAAAU0VMRUNUIDEIAAAAU0VMRUNUIDIVBBk8SARyb290FQQAFQwlABgIcXVlc3Rpb24lAAAVDCUAGANTUUwlAAAWBBkcGSwmCBwVDBkVABkYCHF1ZXN0aW9uFQIWBBZGFkYmCDw2ACgCcTIYAnExABkcFQYVABUCAAAAJk4cFQwZFQAZGANTUUwVAhYEFl4WXiZOPDYAKAhTRUxFQ1QgMhgIU0VMRUNUIDEAGRwVBhUAFQIAAAAWpAEWBAAoCWh5cGFycXVldACxAAAAUEFSMQ==',
} as const;

const blobOf = (base64: string): Blob =>
  new Blob([Uint8Array.from(atob(base64), (char) => char.charCodeAt(0))]);

const file = (path: string, size = 1024): FilesetFileOutput => ({
  path,
  size,
  file_ref: `ref-${path}`,
  file_url: `/apis/files/v2/workspaces/ws/filesets/ds/-/${path}`,
});

const SHARD = /(^|\/)train-\d{5}-of-\d{5}\b[^/]*\.parquet$/;

const params = (overrides = {}) => ({
  workspace: 'ws',
  filesetName: 'ds',
  pattern: SHARD,
  rowCount: 2,
  ...overrides,
});

const axiosError = (status?: number, code?: string): AxiosError => {
  const error = new AxiosError('upstream said no', code);
  if (status !== undefined) error.response = { status } as never;
  return error;
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('matchDatasetFiles', () => {
  it('returns matching shards in split order regardless of listing order', () => {
    const files = [
      file('data/train-00002-of-00003-c.parquet'),
      file('README.md'),
      file('data/train-00000-of-00003-a.parquet'),
      file('data/train-00001-of-00003-b.parquet'),
    ];

    expect(matchDatasetFiles(files, SHARD).map((f) => f.path)).toEqual([
      'data/train-00000-of-00003-a.parquet',
      'data/train-00001-of-00003-b.parquet',
      'data/train-00002-of-00003-c.parquet',
    ]);
  });

  /** The likeliest real failure, so the message has to name both sides of the mismatch. */
  it('names the pattern and the files present when nothing matches', () => {
    expect(() => matchDatasetFiles([file('README.md')], SHARD)).toThrow(
      /No dataset file matched .*train.*The repository holds: README\.md/s
    );
  });

  it('reports an empty repository rather than an empty list', () => {
    expect(() => matchDatasetFiles([], SHARD)).toThrow(/holds: no files/);
  });

  it('refuses a match that is not Parquet', () => {
    expect(() => matchDatasetFiles([file('train-00000-of-00001.json')], /train-.*/)).toThrow(
      /Expected a Parquet dataset file/
    );
  });

  /** A `g` pattern carries lastIndex between calls and would match every other file. */
  it('is not confused by a global pattern', () => {
    const files = [
      file('data/train-00000-of-00002-a.parquet'),
      file('data/train-00001-of-00002-b.parquet'),
    ];
    expect(matchDatasetFiles(files, new RegExp(SHARD.source, 'g'))).toHaveLength(2);
  });
});

describe('readParquetRows', () => {
  it('decodes real Parquet into plain objects', async () => {
    await expect(readParquetRows(blobOf(PARQUET.birdRows), 2)).resolves.toEqual([
      { question: 'q1', SQL: 'SELECT 1' },
      { question: 'q2', SQL: 'SELECT 2' },
    ]);
  });

  it('decodes only the rows asked for', async () => {
    await expect(readParquetRows(blobOf(PARQUET.tenRows), 3)).resolves.toEqual([
      { n: 'row-0' },
      { n: 'row-1' },
      { n: 'row-2' },
    ]);
  });

  it('reports unreadable Parquet as a dataset problem, not a decoder stack trace', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    await expect(readParquetRows(new Blob(['not parquet at all']), 1)).rejects.toThrow(
      /Could not read the dataset file/
    );
    // The decoder's own message is logged rather than shown, so it stays diagnosable.
    expect(consoleError).toHaveBeenCalledWith(
      expect.stringContaining('Failed to decode Parquet dataset file'),
      expect.anything()
    );

    consoleError.mockRestore();
  });
});

describe('fetchFilesetRows', () => {
  it('lists, matches and decodes the shard through the files service', async () => {
    listMock.mockResolvedValue({
      data: [file('README.md'), file('data/train-00000-of-00001-a.parquet')],
    });
    fetchMock.mockResolvedValue(blobOf(PARQUET.twoRows));

    await expect(fetchFilesetRows(params(), new AbortController().signal)).resolves.toEqual([
      { n: 'a' },
      { n: 'b' },
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('stops after the first shard once the row quota is met', async () => {
    listMock.mockResolvedValue({
      data: [
        file('data/train-00000-of-00002-a.parquet'),
        file('data/train-00001-of-00002-b.parquet'),
      ],
    });
    fetchMock.mockResolvedValue(blobOf(PARQUET.tenRows));

    const rows = await fetchFilesetRows(params({ rowCount: 2 }), new AbortController().signal);

    expect(rows).toHaveLength(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reads across shards when one is not enough', async () => {
    listMock.mockResolvedValue({
      data: [
        file('data/train-00000-of-00002-a.parquet'),
        file('data/train-00001-of-00002-b.parquet'),
      ],
    });
    // Both shards hold the same two rows, so the second contributes only the row the first
    // was short of. The point here is the walk across shards, not the values.
    fetchMock.mockResolvedValue(blobOf(PARQUET.twoRows));

    const rows = await fetchFilesetRows(params({ rowCount: 3 }), new AbortController().signal);

    expect(rows).toEqual([{ n: 'a' }, { n: 'b' }, { n: 'a' }]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('reports progress against the total the listing declared', async () => {
    listMock.mockResolvedValue({ data: [file('data/train-00000-of-00001-a.parquet', 500)] });
    fetchMock.mockResolvedValue(blobOf(PARQUET.twoRows));
    const onDownloadProgress = vi.fn();

    await fetchFilesetRows(
      params({ rowCount: 1, onDownloadProgress }),
      new AbortController().signal
    );

    expect(onDownloadProgress).toHaveBeenLastCalledWith(500, 500);
  });

  /**
   * The caps bound what is read, not what is matched. A repo re-sharded into far more (or
   * far larger) files than the limits is fine so long as the rows come from the first shard
   * — which is the whole reason `filePattern` is allowed to tolerate re-sharding.
   */
  it('reads a huge multi-shard repo when the first shard covers the quota', async () => {
    listMock.mockResolvedValue({
      data: Array.from({ length: 40 }, (_, i) =>
        file(`data/train-000${String(i).padStart(2, '0')}-of-00040-x.parquet`, 60 * 1024 * 1024)
      ),
    });
    fetchMock.mockResolvedValue(blobOf(PARQUET.twoRows));

    await expect(
      fetchFilesetRows(params({ rowCount: 2 }), new AbortController().signal)
    ).resolves.toHaveLength(2);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('refuses a shard that would push the download past the byte limit', async () => {
    listMock.mockResolvedValue({
      data: [file('data/train-00000-of-00001-x.parquet', 65 * 1024 * 1024)],
    });

    await expect(fetchFilesetRows(params(), new AbortController().signal)).rejects.toThrow(
      /above the 64MB limit/
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('gives up once reading has walked more shards than the limit', async () => {
    listMock.mockResolvedValue({
      data: Array.from({ length: 9 }, (_, i) => file(`data/train-0000${i}-of-00009-x.parquet`)),
    });
    fetchMock.mockResolvedValue(blobOf(PARQUET.twoRows));

    await expect(
      fetchFilesetRows(params({ rowCount: 50 }), new AbortController().signal)
    ).rejects.toThrow(/more than 8 shards/);
    expect(fetchMock).toHaveBeenCalledTimes(8);
  });

  it('tags a 503 as transient so the query can retry it', async () => {
    listMock.mockRejectedValue(axiosError(503));

    await expect(fetchFilesetRows(params(), new AbortController().signal)).rejects.toThrow(
      TransientFilesetError
    );
  });

  it('leaves a 404 untagged so it fails immediately', async () => {
    listMock.mockRejectedValue(axiosError(404));

    const promise = fetchFilesetRows(params(), new AbortController().signal);
    await expect(promise).rejects.toThrow(AxiosError);
    await expect(promise).rejects.not.toThrow(TransientFilesetError);
  });

  it('does not treat a cancelled request as retryable', async () => {
    listMock.mockRejectedValue(axiosError(undefined, AxiosError.ERR_CANCELED));

    await expect(fetchFilesetRows(params(), new AbortController().signal)).rejects.not.toThrow(
      TransientFilesetError
    );
  });
});

describe('filesetRowsQueryOptions', () => {
  const { retry } = filesetRowsQueryOptions(params());

  it('retries transient failures up to the budget', () => {
    const retryFn = retry as (count: number, error: Error) => boolean;
    expect(retryFn(0, new TransientFilesetError('flaky'))).toBe(true);
    expect(retryFn(1, new TransientFilesetError('flaky'))).toBe(true);
    expect(retryFn(2, new TransientFilesetError('flaky'))).toBe(false);
  });

  it('never retries a non-transient failure', () => {
    const retryFn = retry as (count: number, error: Error) => boolean;
    expect(retryFn(0, new Error('no file matched'))).toBe(false);
  });

  it('keys on the pattern so two splits of one fileset do not share a cache entry', () => {
    const train = filesetRowsQueryOptions(params({ pattern: /train-.*\.parquet$/ })).queryKey;
    const validation = filesetRowsQueryOptions(
      params({ pattern: /validation-.*\.parquet$/ })
    ).queryKey;

    expect(train).not.toEqual(validation);
  });
});
