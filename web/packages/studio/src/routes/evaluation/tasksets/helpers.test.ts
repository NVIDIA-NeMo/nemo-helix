// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  configSection,
  parseFilesRef,
  parseTaskRef,
  tagNames,
} from '@studio/routes/evaluation/tasksets/helpers';

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

it('parseFilesRef splits workspace, fileset, and path', () => {
  expect(parseFilesRef('ws/files#a/b.tar.gz')).toEqual({
    workspace: 'ws',
    fileset: 'files',
    path: 'a/b.tar.gz',
  });
});

it('tagNames puts latest first, then sorts alphabetically', () => {
  expect(tagNames({ stable: 1, beta: 2, latest: 2 })).toEqual(['latest', 'beta', 'stable']);
});

it('configSection returns an empty object for a missing or non-object section', () => {
  expect(configSection({ environment: { cpus: 1 } }, 'environment')).toEqual({ cpus: 1 });
  expect(configSection({ environment: 'x' }, 'environment')).toEqual({});
  expect(configSection(undefined, 'verifier')).toEqual({});
});
