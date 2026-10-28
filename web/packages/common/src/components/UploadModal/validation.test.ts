// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UploadFile } from '@nemo/common/src/components/UploadModal/types';
import { validateUploadForm } from '@nemo/common/src/components/UploadModal/validation';

const readme = {
  id: 'readme',
  type: 'existing',
  file: { path: 'README.md' },
} as unknown as UploadFile;

describe('validateUploadForm file type', () => {
  const base = {
    selectedFiles: [readme],
    selectedDataset: undefined,
    isDatasetRequired: false,
    acceptableFileTypes: ['.json', '.csv'],
  };

  it('rejects an unsupported selection in disable mode', () => {
    expect(validateUploadForm({ ...base, invalidFileMode: 'disable' }).file).toMatch(
      /\.json, \.csv/
    );
  });

  it('leaves type checking to the parent in show mode', () => {
    expect(validateUploadForm({ ...base, invalidFileMode: 'show' })).toEqual({});
  });
});
