// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  filesDownloadFile,
  filesListFilesetFiles,
  filesUploadFile,
} from '@nemo/sdk/generated/platform/files';
import { copyFilesetFiles, uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesDownloadFile: vi.fn(),
  filesListFilesetFiles: vi.fn(),
  filesUploadFile: vi.fn(),
}));

const entries = (count: number) =>
  Array.from({ length: count }, (_, index) => ({
    path: `file-${index}.txt`,
    file: new File(['x'], `file-${index}.txt`),
  }));

afterEach(() => {
  vi.clearAllMocks();
});

describe('uploadFilesetEntries', () => {
  it('uploads every entry', async () => {
    vi.mocked(filesUploadFile).mockResolvedValue({} as never);

    await uploadFilesetEntries('ws', 'bundle', entries(10));

    expect(
      vi
        .mocked(filesUploadFile)
        .mock.calls.map((call) => call[2])
        .sort()
    ).toEqual(
      entries(10)
        .map((entry) => entry.path)
        .sort()
    );
  });

  it('stops the other workers once one upload fails', async () => {
    vi.mocked(filesUploadFile)
      .mockRejectedValueOnce(new Error('quota exceeded'))
      .mockResolvedValue({} as never);

    await expect(uploadFilesetEntries('ws', 'bundle', entries(20))).rejects.toThrow(
      'quota exceeded'
    );

    expect(vi.mocked(filesUploadFile).mock.calls.length).toBeLessThanOrEqual(6);
  });

  it('waits for in-flight uploads to settle before rejecting', async () => {
    let finishSlowUpload!: () => void;
    vi.mocked(filesUploadFile)
      .mockImplementationOnce(
        () => new Promise((resolve) => (finishSlowUpload = () => resolve({} as never)))
      )
      .mockRejectedValueOnce(new Error('quota exceeded'))
      .mockResolvedValue({} as never);

    let settled = false;
    const upload = uploadFilesetEntries('ws', 'bundle', entries(2)).catch((error: unknown) => {
      settled = true;
      throw error;
    });
    await vi.waitFor(() => expect(filesUploadFile).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(settled).toBe(false);

    finishSlowUpload();
    await expect(upload).rejects.toThrow('quota exceeded');
  });
});

describe('copyFilesetFiles', () => {
  it('uploads every file of the source fileset under the same path', async () => {
    vi.mocked(filesListFilesetFiles).mockResolvedValue({
      data: [{ path: 'agent.yaml' }, { path: 'skills/search.md' }],
    } as never);
    vi.mocked(filesDownloadFile).mockImplementation(
      async (_workspace, _name, path) => new Blob([`contents of ${path}`])
    );
    vi.mocked(filesUploadFile).mockResolvedValue({} as never);

    await copyFilesetFiles(
      { workspace: 'src', name: 'a-ethos' },
      { workspace: 'ws', name: 'b-ethos' }
    );

    expect(vi.mocked(filesDownloadFile).mock.calls.map((call) => call.slice(0, 3))).toEqual([
      ['src', 'a-ethos', 'agent.yaml'],
      ['src', 'a-ethos', 'skills/search.md'],
    ]);
    const uploads = vi.mocked(filesUploadFile).mock.calls;
    expect(uploads.map((call) => call.slice(0, 3))).toEqual([
      ['ws', 'b-ethos', 'agent.yaml'],
      ['ws', 'b-ethos', 'skills/search.md'],
    ]);
    expect(await uploads[1]![3].text()).toBe('contents of skills/search.md');
  });
});
