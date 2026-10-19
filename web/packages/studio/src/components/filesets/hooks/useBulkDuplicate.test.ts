// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import { useDatasetFilesUpload } from '@studio/api/datasets/useDatasetFilesUpload';
import { useBulkDuplicate } from '@studio/components/filesets/hooks/useBulkDuplicate';
import { useDownloadFileAsArrayBuffer } from '@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { act, renderHook, waitFor } from '@testing-library/react';

vi.mock('@nemo/common/src/providers/toast/useToast');
vi.mock('@nemo/sdk/generated/platform/files');
vi.mock('@studio/api/datasets/useDatasetFilesUpload');
vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer');

const mockUseToast = vi.mocked(useToast);
const mockFilesListFilesetFiles = vi.mocked(filesListFilesetFiles);
const mockUseDatasetFilesUpload = vi.mocked(useDatasetFilesUpload);
const mockUseDownloadFileAsArrayBuffer = vi.mocked(useDownloadFileAsArrayBuffer);

const workspace = 'ws';
const datasetName = 'ds';

const file = (path: string): FileSystemFile => ({
  type: 'file',
  path,
  size: 1,
  oid: path,
});

// Minimal stand-in for the SDK's `ListFilesetFilesResponse`. The hook only
// reads `data[].path`, so we don't need to satisfy the full FilesetFileOutput
// shape here.
const listResponse = (paths: string[]) =>
  ({ data: paths.map((p) => ({ path: p })) }) as unknown as Awaited<
    ReturnType<typeof filesListFilesetFiles>
  >;

describe('useBulkDuplicate', () => {
  const workingWithId = vi.fn();
  const dismissToast = vi.fn();
  const success = vi.fn();
  const error = vi.fn();
  const uploadMutateAsync = vi.fn();
  const downloadAsArrayBuffer = vi.fn();
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    workingWithId.mockReturnValue('toast-id');

    mockUseToast.mockReturnValue({
      workingWithId,
      dismissToast,
      success,
      error,
      info: vi.fn(),
      warning: vi.fn(),
    } as unknown as ReturnType<typeof useToast>);

    mockUseDatasetFilesUpload.mockReturnValue({
      mutateAsync: uploadMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useDatasetFilesUpload>);

    mockUseDownloadFileAsArrayBuffer.mockReturnValue(downloadAsArrayBuffer);

    // Default: directory is empty on the server. Individual tests override.
    mockFilesListFilesetFiles.mockResolvedValue(listResponse([]));
    uploadMutateAsync.mockResolvedValue(undefined);
    downloadAsArrayBuffer.mockResolvedValue(new ArrayBuffer(8));

    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  const render = () => renderHook(() => useBulkDuplicate({ workspace, datasetName }));

  const getUploadedNames = () => {
    const uploadedFiles = uploadMutateAsync.mock.calls[0][0].files as File[];
    return uploadedFiles.map((f) => f.name);
  };

  it('is a no-op for an empty array and resolves to true (nothing failed)', async () => {
    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([]);
    });

    expect(outcome).toBe(true);
    expect(workingWithId).not.toHaveBeenCalled();
    expect(mockFilesListFilesetFiles).not.toHaveBeenCalled();
    expect(downloadAsArrayBuffer).not.toHaveBeenCalled();
    expect(uploadMutateAsync).not.toHaveBeenCalled();
  });

  it('picks "<name>-copy.<ext>" when no collision exists and resolves to true', async () => {
    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([file('data/a.txt')]);
    });

    expect(outcome).toBe(true);
    expect(getUploadedNames()).toEqual(['data/a-copy.txt']);
    expect(success).toHaveBeenCalledWith('File duplicated successfully');
  });

  it('bumps the suffix to "-copy-2" when "-copy" is already taken on the server', async () => {
    mockFilesListFilesetFiles.mockResolvedValue(listResponse(['data/a.txt', 'data/a-copy.txt']));

    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('data/a.txt')]);
    });

    expect(getUploadedNames()).toEqual(['data/a-copy-2.txt']);
  });

  it('keeps bumping past multiple existing copies', async () => {
    mockFilesListFilesetFiles.mockResolvedValue(
      listResponse(['data/a.txt', 'data/a-copy.txt', 'data/a-copy-2.txt'])
    );

    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('data/a.txt')]);
    });

    expect(getUploadedNames()).toEqual(['data/a-copy-3.txt']);
  });

  it('gives siblings in the same batch distinct "-copy" suffixes without re-listing', async () => {
    mockFilesListFilesetFiles.mockResolvedValue(listResponse(['foo/a.txt']));

    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('foo/a.txt'), file('foo/a.txt')]);
    });

    // Only one listing per unique directory, even for two sources in the same dir.
    expect(mockFilesListFilesetFiles).toHaveBeenCalledTimes(1);
    expect(getUploadedNames()).toEqual(['foo/a-copy.txt', 'foo/a-copy-2.txt']);
  });

  it('lists each unique directory exactly once and isolates per-dir reserved sets', async () => {
    mockFilesListFilesetFiles.mockImplementation(async (_ws, _ds, params) => {
      if (params?.path === 'foo/') return listResponse(['foo/a.txt', 'foo/a-copy.txt']);
      if (params?.path === 'bar/') return listResponse(['bar/b.txt']);
      return listResponse([]);
    });

    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('foo/a.txt'), file('bar/b.txt')]);
    });

    expect(mockFilesListFilesetFiles).toHaveBeenCalledTimes(2);
    // foo/ already has -copy → bumps to -copy-2; bar/ is clean → -copy.
    expect(getUploadedNames().sort()).toEqual(['bar/b-copy.txt', 'foo/a-copy-2.txt']);
  });

  it('uses no directory prefix when duplicating a file at the fileset root', async () => {
    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('a.txt')]);
    });

    expect(mockFilesListFilesetFiles).toHaveBeenCalledWith(workspace, datasetName, {
      path: undefined,
    });
    expect(getUploadedNames()).toEqual(['a-copy.txt']);
  });

  it('handles files without an extension', async () => {
    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('README')]);
    });

    expect(getUploadedNames()).toEqual(['README-copy']);
  });

  it('resolves to false and reports a partial-success toast when some downloads fail', async () => {
    downloadAsArrayBuffer.mockResolvedValueOnce(new ArrayBuffer(4)).mockResolvedValueOnce(null);

    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([file('a.txt'), file('b.txt')]);
    });

    expect(outcome).toBe(false);
    const uploadedFiles = uploadMutateAsync.mock.calls[0][0].files as File[];
    expect(uploadedFiles).toHaveLength(1);
    expect(error).toHaveBeenCalledWith('Duplicated 1 of 2 files. 1 failed.');
    expect(success).not.toHaveBeenCalled();
  });

  it('resolves to false when every download fails', async () => {
    downloadAsArrayBuffer.mockResolvedValue(null);

    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([file('a.txt'), file('b.txt')]);
    });

    expect(outcome).toBe(false);
    expect(uploadMutateAsync).not.toHaveBeenCalled();
    expect(error).toHaveBeenCalledWith('Failed to duplicate files');
  });

  it('keeps isDuplicating true for the full list -> download -> upload flow', async () => {
    let resolveList: (value: Awaited<ReturnType<typeof filesListFilesetFiles>>) => void = () => {};
    let resolveDownload: (value: ArrayBuffer | null) => void = () => {};
    let resolveUpload: (value: unknown) => void = () => {};

    mockFilesListFilesetFiles.mockReturnValue(
      new Promise((res) => {
        resolveList = res;
      })
    );
    downloadAsArrayBuffer.mockReturnValue(
      new Promise((res) => {
        resolveDownload = res;
      })
    );
    uploadMutateAsync.mockReturnValue(
      new Promise((res) => {
        resolveUpload = res;
      })
    );

    const { result } = render();

    expect(result.current.isDuplicating).toBe(false);

    let pending: Promise<boolean> | undefined;
    act(() => {
      pending = result.current.handleBulkDuplicate([file('a.txt')]);
    });

    // Listing phase.
    await waitFor(() => expect(result.current.isDuplicating).toBe(true));

    // Unblock listing → download phase; still busy.
    await act(async () => {
      resolveList(listResponse([]));
    });
    expect(result.current.isDuplicating).toBe(true);

    // Unblock download → upload phase; still busy.
    await act(async () => {
      resolveDownload(new ArrayBuffer(1));
    });
    expect(result.current.isDuplicating).toBe(true);

    // Unblock upload → flow completes and the flag clears.
    await act(async () => {
      resolveUpload(undefined);
      await pending;
    });
    expect(result.current.isDuplicating).toBe(false);
  });

  it('resolves to false, logs, and surfaces an error toast when listing fails', async () => {
    const listError = new Error('list exploded');
    mockFilesListFilesetFiles.mockRejectedValue(listError);

    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([file('a.txt')]);
    });

    expect(outcome).toBe(false);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Bulk duplicate failed', listError);
    expect(dismissToast).toHaveBeenCalledWith('toast-id');
    expect(error).toHaveBeenCalledWith('Failed to duplicate file');
    expect(downloadAsArrayBuffer).not.toHaveBeenCalled();
    expect(uploadMutateAsync).not.toHaveBeenCalled();
    expect(result.current.isDuplicating).toBe(false);
  });

  it('resolves to false, logs, and surfaces an error toast when upload throws', async () => {
    const uploadError = new Error('upload exploded');
    uploadMutateAsync.mockRejectedValue(uploadError);

    const { result } = render();

    let outcome: boolean | undefined;
    await act(async () => {
      outcome = await result.current.handleBulkDuplicate([file('a.txt')]);
    });

    expect(outcome).toBe(false);
    expect(consoleErrorSpy).toHaveBeenCalledWith('Bulk duplicate failed', uploadError);
    expect(error).toHaveBeenCalledWith('Failed to duplicate file');
    expect(result.current.isDuplicating).toBe(false);
  });

  it('uses plural error copy for multi-file batches', async () => {
    mockFilesListFilesetFiles.mockRejectedValue(new Error('boom'));

    const { result } = render();

    await act(async () => {
      await result.current.handleBulkDuplicate([file('a.txt'), file('b.txt')]);
    });

    expect(error).toHaveBeenCalledWith('Failed to duplicate files');
  });
});
