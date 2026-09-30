// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesUploadFile } from '@nemo/sdk/generated/platform/files';
import { uploadFilesetEntries } from '@studio/api/files/uploadFilesetEntries';

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
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
});
