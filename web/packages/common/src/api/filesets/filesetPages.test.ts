// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getFilesetQueryKey,
  getFilesetsQueryKey,
} from '@nemo/common/src/api/filesets/filesetPages';
import {
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';

describe('fileset query keys', () => {
  it('match the SDK keys that Studio invalidates', () => {
    expect(getFilesetsQueryKey('ws')).toEqual(getFilesListFilesetsQueryKey('ws'));
    expect(getFilesetQueryKey('ws', 'data')).toEqual(getFilesRetrieveFilesetQueryKey('ws', 'data'));
  });
});
