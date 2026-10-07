// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilesetFile } from '@studio/api/files/useFilesetFile';
import { useFetchFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import { wrapper } from '@studio/tests/util/TestQueryClient';
import { renderHook, waitFor } from '@testing-library/react';

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer');

const fetchFile = vi.fn();

beforeEach(() => {
  vi.mocked(useFetchFileAsArrayBuffer).mockReturnValue(fetchFile);
  fetchFile.mockImplementation(async () => new TextEncoder().encode('v1').buffer);
});

afterEach(() => {
  vi.clearAllMocks();
});

const renderFilesetFile = (path: string) =>
  renderHook(
    (props: { path: string }) => useFilesetFile({ workspace: 'ws', fileset: 'fs', ...props }),
    {
      wrapper,
      initialProps: { path },
    }
  );

describe('useFilesetFile', () => {
  it('names the downloaded file after the last path segment', async () => {
    const { result } = renderFilesetFile('output/part-0.parquet');

    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data?.name).toBe('part-0.parquet');
    expect(await result.current.data?.text()).toBe('v1');
    expect(fetchFile).toHaveBeenCalledWith({
      workspace: 'ws',
      datasetName: 'fs',
      path: 'output/part-0.parquet',
    });
  });

  it('downloads current contents when a path is selected again', async () => {
    const { result, rerender } = renderFilesetFile('a.csv');
    await waitFor(() => expect(result.current.data).toBeDefined());

    rerender({ path: 'b.csv' });
    await waitFor(() => expect(result.current.data?.name).toBe('b.csv'));
    fetchFile.mockImplementation(async () => new TextEncoder().encode('v2').buffer);
    rerender({ path: 'a.csv' });

    await waitFor(() => expect(result.current.data?.name).toBe('a.csv'));
    expect(await result.current.data?.text()).toBe('v2');
  });

  it('surfaces the download error', async () => {
    fetchFile.mockRejectedValue(new Error('Unable to find base file.'));
    const { result } = renderFilesetFile('a.csv');

    await waitFor(() => expect(result.current.error?.message).toBe('Unable to find base file.'));
  });

  it('waits for a path before downloading', () => {
    renderFilesetFile('');

    expect(fetchFile).not.toHaveBeenCalled();
  });
});
