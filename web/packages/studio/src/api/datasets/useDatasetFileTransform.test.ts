// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import { useDatasetFileTransform } from '@studio/api/datasets/useDatasetFileTransform';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesUploadFile: vi.fn().mockResolvedValue({ path: 'data.jsonl' }),
}));

vi.mock('@studio/api/datasets/invalidateDatasetCaches', () => ({
  invalidateDatasetCaches: vi.fn(),
}));

const uploadMock = vi.mocked(filesUploadFile);

const baseVariables = {
  workspace: 'default',
  datasetName: 'test',
  filepath: 'data.jsonl',
  template: { id: '{{ task_id }}' },
  fileContent: '{"task_id":"a1"}',
};

const renderTransform = () =>
  renderHook(() => useDatasetFileTransform({}), { wrapper: TestProviders });

describe('useDatasetFileTransform', () => {
  beforeEach(() => {
    uploadMock.mockClear();
  });

  it('uploads the remapped rows for a JSONL file', async () => {
    const { result } = renderTransform();

    await result.current.mutateAsync(baseVariables);

    expect(uploadMock).toHaveBeenCalledTimes(1);
  });

  it('gives every row its own identifier for a generated column', async () => {
    const { result } = renderTransform();

    await result.current.mutateAsync({
      ...baseVariables,
      template: { id: '{{ row_id }}' },
      fileContent: '{"a":1}\n{"a":2}',
      generatedIdColumn: 'row_id',
    });

    const blob = uploadMock.mock.calls[0][3] as Blob;
    const ids = (await blob.text()).split('\n').map((line) => JSON.parse(line).id);
    expect(new Set(ids).size).toBe(2);
    // A full UUID, not a truncated slice — the transform can run over large files.
    expect(ids[0]).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
  });

  it('does not upload anything when the file has no rows', async () => {
    const { result } = renderTransform();

    await expect(
      result.current.mutateAsync({ ...baseVariables, fileContent: '  \n  ' })
    ).rejects.toThrow(/No rows could be read/);

    expect(uploadMock).not.toHaveBeenCalled();
  });

  it('does not upload anything when every row fails to parse', async () => {
    // `parseFileContent` logs each unparseable line.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { result } = renderTransform();

    await expect(
      result.current.mutateAsync({ ...baseVariables, fileContent: 'not json at all' })
    ).rejects.toThrow(/could not be parsed/);

    expect(uploadMock).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('does not upload a partially parsed file', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { result } = renderTransform();

    await expect(
      result.current.mutateAsync({
        ...baseVariables,
        fileContent: '{"task_id":"a1"}\nnot json at all\n{"task_id":"a2"}',
      })
    ).rejects.toThrow(/1 line\(s\) could not be parsed/);

    expect(uploadMock).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('refuses to overwrite a file that is not JSONL', async () => {
    const { result } = renderTransform();

    await expect(
      result.current.mutateAsync({
        ...baseVariables,
        filepath: 'data.csv',
        fileContent: 'task_id\na1',
      })
    ).rejects.toThrow(/Only JSONL files/);

    expect(uploadMock).not.toHaveBeenCalled();
  });
});
