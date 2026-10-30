// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getFilesListFilesetFilesQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { useFilesetFile } from '@studio/api/files/useFilesetFile';
import {
  type MessageSelector,
  messagesPath,
  messagesShape,
  useDatasetPreview,
} from '@studio/components/evaluation/EvalWizard/useDatasetPreview';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { PARQUET, parquetFile } from '@studio/tests/util/parquetFixtures';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import { renderHook, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';

const downloadFileHead = vi.hoisted(() => vi.fn());
const fetchFile = vi.hoisted(() => vi.fn());

vi.mock('@studio/components/filesets/hooks/useDownloadFileHead', () => ({
  useDownloadFileHead: () => downloadFileHead,
}));

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer: () => fetchFile,
}));

/** Mirrors ``extractUserFriendlyKeysFromRow``: `null` is a contentless message
 *  it drops, which still consumes an index. */
const selectors = (...roles: (string | null)[]): MessageSelector[] =>
  roles.flatMap((role, index) =>
    role === null ? [] : [{ label: role, role, selector: `conversation[${index}].content` }]
  );

describe('messagesPath', () => {
  it('selects the last turn with the role', () => {
    expect(messagesPath('conversation', 'user')).toBe('conversation[role=user].content');
    expect(messagesPath('conversation', 'assistant')).toBe('conversation[role=assistant].content');
  });
});

describe('messagesShape', () => {
  it.each([
    ['a single-turn exchange', ['user', 'assistant'], { input: true, reference: true }],
    [
      'a multi-turn conversation',
      ['user', 'assistant', 'user', 'assistant'],
      { input: true, reference: true },
    ],
    ['a leading system message', ['system', 'user', 'assistant'], { input: true, reference: true }],
    // The last user turn is unanswered, so the earlier reply is not its reference.
    ['a trailing user turn', ['user', 'assistant', 'user'], { input: true, reference: false }],
    ['no assistant turn', ['user'], { input: true, reference: false }],
    ['a prompts-only file', ['user', 'user'], { input: true, reference: false }],
    ['no user turn', ['system', 'assistant'], { input: false, reference: false }],
    // A contentless turn (e.g. a tool call) between the two does not break the pair.
    [
      'a contentless turn in between',
      ['user', null, 'assistant'],
      { input: true, reference: true },
    ],
    // A contentless assistant turn is not a reference.
    ['a contentless last reply', ['user', null], { input: true, reference: false }],
    ['no messages', [], { input: false, reference: false }],
  ] as const)('handles %s', (_, roles, expected) => {
    expect(messagesShape(selectors(...roles))).toEqual(expected);
  });
});

describe('useDatasetPreview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('reads a Parquet file whole and offers its columns', async () => {
    // Listed larger than the 64 KB head a line-delimited file is read from.
    const listedSize = 100_000;
    server.use(
      http.get(mockApiUrl(getFilesRetrieveFilesetQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({ name: 'data', workspace: 'ws' })
      ),
      http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({ data: [{ path: 'rows.parquet', size: listedSize }] })
      )
    );
    fetchFile.mockImplementation(() => parquetFile(PARQUET.twoRows).arrayBuffer());

    const { result } = renderHook(() => useDatasetPreview('ws/data#rows.parquet'), { wrapper });

    await waitFor(() => expect(result.current.row).toEqual({ prompt: 'hi', expected: 'hi' }));
    expect(result.current.keyOptions.map((option) => option.value)).toEqual(['prompt', 'expected']);
    expect(result.current.rowCount).toBe(2);
    expect(result.current.error).toBeNull();
    expect(fetchFile).toHaveBeenCalledWith(expect.objectContaining({ path: 'rows.parquet' }));
    expect(downloadFileHead).not.toHaveBeenCalled();
  });

  it('shares one download with a whole-file read of the same file', async () => {
    server.use(
      http.get(mockApiUrl(getFilesRetrieveFilesetQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({ name: 'data', workspace: 'ws' })
      ),
      http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
        HttpResponse.json({ data: [{ path: 'rows.parquet', size: 100_000 }] })
      )
    );
    fetchFile.mockImplementation(() => parquetFile(PARQUET.twoRows).arrayBuffer());

    const { result } = renderHook(
      () => ({
        preview: useDatasetPreview('ws/data#rows.parquet'),
        file: useFilesetFile({ workspace: 'ws', fileset: 'data', path: 'rows.parquet' }),
      }),
      { wrapper }
    );

    await waitFor(() => expect(result.current.preview.row).not.toBeNull());
    expect(result.current.file.data).toBeInstanceOf(File);
    expect(fetchFile).toHaveBeenCalledTimes(1);
  });
});
