// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useFilePreview } from '@nemo/common/src/components/DatasetFileSelect/hooks/useFilePreview';
import type { FileListItem } from '@nemo/common/src/components/FileList';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';

const downloadFileMock = vi.hoisted(() => vi.fn());

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>();
  return { ...actual, filesDownloadFile: downloadFileMock };
});

const makeWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const fileWithDataset: FileListItem = {
  dataset: {
    id: 'my-workspace/my-dataset',
    name: 'my-dataset',
    workspace: 'my-workspace',
    description: '',
    purpose: 'dataset',
    storage: { type: 'local', path: '/data' },
    metadata: {},
    custom_fields: {},
    project: 'default',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  path: 'train.jsonl',
  url: 'fileset://my-workspace/my-dataset/train.jsonl',
};

const fileWithUrlOnly: FileListItem = {
  path: 'train.jsonl',
  url: 'fileset://my-workspace/my-dataset/train.jsonl',
};

const fileWithLocalContent: FileListItem = {
  path: 'local.jsonl',
  content: '{"foo":"bar"}',
};

describe('useFilePreview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('starts with no preview file', () => {
    const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });
    expect(result.current.previewFile).toBeNull();
    expect(result.current.previewContent).toBeUndefined();
    expect(result.current.isLoadingPreview).toBe(false);
    expect(result.current.previewError).toBeNull();
  });

  describe('local content', () => {
    it('returns local content without fetching from remote', async () => {
      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithLocalContent);
      });

      await waitFor(() => expect(result.current.previewContent).toBe('{"foo":"bar"}'));
      expect(downloadFileMock).not.toHaveBeenCalled();
    });
  });

  describe('remote fetch with dataset object', () => {
    it('calls filesDownloadFile with workspace and name from dataset', async () => {
      downloadFileMock.mockResolvedValue(new Blob(['remote content'], { type: 'text/plain' }));

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithDataset);
      });

      await waitFor(() => expect(result.current.previewContent).toBe('remote content'));
      expect(downloadFileMock).toHaveBeenCalledWith('my-workspace', 'my-dataset', 'train.jsonl');
    });
  });

  describe('remote fetch with URL only (no dataset object)', () => {
    it('parses workspace and name from fileset URL when dataset is absent', async () => {
      downloadFileMock.mockResolvedValue(new Blob(['url-only content'], { type: 'text/plain' }));

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithUrlOnly);
      });

      await waitFor(() => expect(result.current.previewContent).toBe('url-only content'));
      expect(downloadFileMock).toHaveBeenCalledWith('my-workspace', 'my-dataset', 'train.jsonl');
    });

    it('throws Missing dataset error when there is no url and no dataset', async () => {
      const noUrlFile: FileListItem = {
        path: 'file.jsonl',
      };

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(noUrlFile);
      });

      await waitFor(() => expect(result.current.previewError).not.toBeNull());
      expect(result.current.previewError?.message).toBe('Missing dataset');
      expect(downloadFileMock).not.toHaveBeenCalled();
    });
  });

  describe('fetch errors', () => {
    it('returns error when filesDownloadFile returns null', async () => {
      downloadFileMock.mockResolvedValue(null as unknown as Blob);

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithDataset);
      });

      await waitFor(() => expect(result.current.previewError).not.toBeNull());
      expect(result.current.previewError?.message).toBe('Failed to fetch file content');
    });

    it('returns error when filesDownloadFile rejects', async () => {
      downloadFileMock.mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithDataset);
      });

      await waitFor(() => expect(result.current.previewError).not.toBeNull());
      expect(result.current.previewError?.message).toBe('Network error');
    });
  });

  describe('clearPreview', () => {
    it('resets preview file to null', async () => {
      downloadFileMock.mockResolvedValue(new Blob(['content'], { type: 'text/plain' }));

      const { result } = renderHook(() => useFilePreview(), { wrapper: makeWrapper() });

      await act(async () => {
        result.current.setPreviewFile(fileWithDataset);
      });
      await waitFor(() => expect(result.current.previewFile).not.toBeNull());

      await act(async () => {
        result.current.clearPreview();
      });

      expect(result.current.previewFile).toBeNull();
    });
  });
});
