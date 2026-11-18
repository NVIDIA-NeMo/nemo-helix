// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseTaskRef } from '@studio/routes/evaluation/taskHelpers';

describe('parseTaskRef', () => {
  it.each([
    ['sort-ints', { workspace: 'default', name: 'sort-ints', version: undefined }],
    ['other/sort-ints', { workspace: 'other', name: 'sort-ints', version: undefined }],
    ['other/sort-ints#abc', { workspace: 'other', name: 'sort-ints', version: 'abc' }],
    ['sort-ints#stable', { workspace: 'default', name: 'sort-ints', version: 'stable' }],
  ])('parses %s', (ref, expected) => {
    expect(parseTaskRef(ref, 'default')).toEqual(expected);
  });
});
