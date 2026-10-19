// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  datasetFileContentQueryOptions,
  EDITOR_MAX_BYTES,
} from '@studio/api/datasets/useDatasetFileContent';

const mocks = vi.hoisted(() => ({
  platformFetch: vi.fn(),
}));

vi.mock('@nemo/sdk/src/utils/platformRequest', () => ({
  platformFetch: mocks.platformFetch,
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>();
  return {
    ...actual,
    filesHeadFile: vi.fn().mockResolvedValue(undefined),
    filesDownloadFile: vi.fn().mockResolvedValue(new Blob(['# heading'])),
  };
});

// customFetch now handles Range requests for text files. Return a Blob that
// resolves to the expected text so the .text() call in the queryFn succeeds.
vi.mock('@nemo/sdk/generated/fetchers/platform', () => ({
  customFetch: vi.fn().mockResolvedValue(new Blob(['# heading'])),
}));

vi.mock('hyparquet', () => ({
  parquetRead: vi.fn(
    async (options: { onComplete?: (rows: Record<string, unknown>[]) => void }) => {
      options.onComplete?.([{ id: 9007199254740993n, label: 'large-int' }]);
    }
  ),
}));

describe('useDatasetFileContent gate', () => {
  const baseParams = { workspace: 'ws', name: 'ds', path: 'README.md' };

  beforeEach(() => {
    mocks.platformFetch.mockReset();
    mocks.platformFetch.mockResolvedValue(
      new Response(null, { headers: { 'content-length': String(EDITOR_MAX_BYTES - 1) } })
    );
  });

  it('allows .md files and returns content via Range fetch', async () => {
    const { queryFn } = datasetFileContentQueryOptions(baseParams);
    await expect((queryFn as () => Promise<string>)()).resolves.toBe('# heading');
  });

  it('rejects files in the binary extension blocklist', async () => {
    // .png is in BINARY_FILE_EXTENSIONS — should throw before any network call.
    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, path: 'image.png' });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(
      /Text preview not available for binary files/
    );
  });

  it('allows files with no extension (treated as text)', async () => {
    // Files with no extension (Makefile, LICENSE, etc.) are not in the blocklist
    // and fall through to the Range fetch — treat as text.
    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, path: 'noextension' });
    await expect((queryFn as () => Promise<string>)()).resolves.toBe('# heading');
  });

  it('returns full text (no preview cap) for fullContent editor loads within the ceiling', async () => {
    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).resolves.toBe('# heading');
  });

  it('reports a missing base file when the HEAD check returns an error status', async () => {
    mocks.platformFetch.mockResolvedValueOnce(new Response(null, { status: 404 }));

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/Unable to find base file/i);
  });

  it('refuses fullContent loads above the editor ceiling instead of truncating', async () => {
    mocks.platformFetch.mockResolvedValueOnce(
      new Response(null, { headers: { 'content-length': String(EDITOR_MAX_BYTES + 1) } })
    );

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
  });

  it('fails closed on fullContent loads when Content-Length is missing', async () => {
    mocks.platformFetch.mockResolvedValueOnce(new Response(null));

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
  });

  it('fails closed on fullContent loads when Content-Length is non-numeric', async () => {
    mocks.platformFetch.mockResolvedValueOnce(
      new Response(null, { headers: { 'content-length': 'not-a-number' } })
    );

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
  });

  it('fails closed on fullContent loads when Content-Length is partially numeric', async () => {
    // parseInt('123garbage') === 123, which would slip a truncated size past the cap.
    mocks.platformFetch.mockResolvedValueOnce(
      new Response(null, { headers: { 'content-length': `${EDITOR_MAX_BYTES - 1}garbage` } })
    );

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
  });

  it('fails closed on fullContent loads when Content-Length is negative', async () => {
    mocks.platformFetch.mockResolvedValueOnce(
      new Response(null, { headers: { 'content-length': '-1' } })
    );

    const { queryFn } = datasetFileContentQueryOptions({ ...baseParams, fullContent: true });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
  });

  it('enforces the cap before downloading a parquet blob on fullContent loads', async () => {
    const { filesDownloadFile } = await import('@nemo/sdk/generated/platform/files');
    vi.mocked(filesDownloadFile).mockClear();
    mocks.platformFetch.mockResolvedValueOnce(new Response(null));

    const { queryFn } = datasetFileContentQueryOptions({
      ...baseParams,
      path: 'data/sample.parquet',
      fullContent: true,
    });
    await expect((queryFn as () => Promise<string>)()).rejects.toThrow(/too large to edit/i);
    expect(filesDownloadFile).not.toHaveBeenCalled();
  });

  it('serializes parquet rows with BigInt columns as JSONL text', async () => {
    const { filesDownloadFile } = await import('@nemo/sdk/generated/platform/files');
    vi.mocked(filesDownloadFile).mockResolvedValueOnce(new Blob(['parquet-bytes']));

    const { queryFn } = datasetFileContentQueryOptions({
      ...baseParams,
      path: 'data/sample.parquet',
    });

    await expect((queryFn as () => Promise<string>)()).resolves.toBe(
      '{"id":"9007199254740993","label":"large-int"}\n'
    );
  });
});
