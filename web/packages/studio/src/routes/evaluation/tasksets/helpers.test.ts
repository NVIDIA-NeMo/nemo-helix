// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';
import {
  configSection,
  matchesTaskFilters,
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

describe('matchesTaskFilters', () => {
  const task: Task = {
    id: 'task-sort-ints',
    name: 'sort-ints',
    workspace: 'ws',
    revision: 1,
    metadata: [{ key: 'difficulty', value: 'easy' }],
    created_at: '2026-10-01T09:00:00Z',
    updated_at: '2026-10-01T09:00:00Z',
    spec: {
      kind: 'harbor',
      native_task_id: 'sort-ints',
      source: {
        kind: 'fileset-archive',
        fileset_ref: 'ws/files#sort-ints.tar.gz',
        files_hash: 'a',
      },
      harbor_hash: { digest: 'a', harbor_version: '0.1.0' },
    },
  };
  const row = { id: 'r', ref: 'ws/sort-ints', workspace: 'ws', name: 'sort-ints', task };
  const unresolved = { ...row, task: undefined };

  it.each([
    [{}, true],
    [{ name: 'SORT' }, true],
    [{ name: 'phish' }, false],
    [{ runner: {} }, true],
    [{ runner: { harbor: true } }, true],
    [{ runner: { evaluator: true } }, false],
    [{ metadata: 'difficulty: EASY' }, true],
    [{ metadata: 'hard' }, false],
  ] as const)('matches %o -> %s', (filters, expected) => {
    expect(matchesTaskFilters(row, filters)).toBe(expected);
  });

  it('drops an unresolved task under runner or metadata filters', () => {
    expect(matchesTaskFilters(unresolved, { name: 'sort' })).toBe(true);
    expect(matchesTaskFilters(unresolved, { runner: { harbor: true } })).toBe(false);
    expect(matchesTaskFilters(unresolved, { metadata: 'easy' })).toBe(false);
  });
});
