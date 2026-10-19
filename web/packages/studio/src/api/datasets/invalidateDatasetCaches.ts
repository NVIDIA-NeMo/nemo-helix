// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getFilesListFilesetFilesQueryKey,
  getFilesListFilesetsQueryKey,
  getFilesRetrieveFilesetQueryKey,
} from '@nemo/sdk/generated/platform/files';
import { queryClient } from '@studio/api/queryClient';

type Scope = 'list' | 'detail' | 'files' | 'content';

export const getDatasetFileContentQueryKey = (workspace: string, name: string, path?: string) =>
  path ? ['fileset-content', workspace, name, path] : ['fileset-content', workspace, name];

/**
 * Invalidates dataset (fileset) query caches using the SDK query keys.
 * Use after mutations that modify filesets or their files.
 * Pass `path` to also invalidate the cached content of a specific file
 * (required when `scopes` includes `'content'`).
 */
export function invalidateDatasetCaches(
  workspace: string,
  name?: string,
  scopes: Scope[] = ['list', 'detail', 'files'],
  path?: string
): Promise<void[]> {
  const ops: Promise<void>[] = [];
  if (scopes.includes('list')) {
    ops.push(queryClient.invalidateQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) }));
  }
  if (scopes.includes('detail') && name) {
    ops.push(
      queryClient.invalidateQueries({
        queryKey: getFilesRetrieveFilesetQueryKey(workspace, name),
      })
    );
  }
  if (scopes.includes('files') && name) {
    ops.push(
      queryClient.invalidateQueries({
        queryKey: getFilesListFilesetFilesQueryKey(workspace, name),
      })
    );
  }
  if (scopes.includes('content') && name) {
    ops.push(
      queryClient.invalidateQueries({
        queryKey: getDatasetFileContentQueryKey(workspace, name, path),
      })
    );
  }
  return Promise.all(ops);
}

/**
 * Resets dataset (fileset) query caches using the SDK query keys.
 * Like invalidate, but also removes cached data (useful after creates/deletes).
 * Pass `path` to also reset the cached content of a specific file
 * (required when `scopes` includes `'content'`).
 */
export function resetDatasetCaches(
  workspace: string,
  name?: string,
  scopes: Scope[] = ['list', 'detail', 'files'],
  path?: string
): Promise<void[]> {
  const ops: Promise<void>[] = [];
  if (scopes.includes('list')) {
    ops.push(queryClient.resetQueries({ queryKey: getFilesListFilesetsQueryKey(workspace) }));
  }
  if (scopes.includes('detail') && name) {
    ops.push(
      queryClient.resetQueries({ queryKey: getFilesRetrieveFilesetQueryKey(workspace, name) })
    );
  }
  if (scopes.includes('files') && name) {
    ops.push(
      queryClient.resetQueries({ queryKey: getFilesListFilesetFilesQueryKey(workspace, name) })
    );
  }
  if (scopes.includes('content') && name) {
    ops.push(
      queryClient.resetQueries({
        queryKey: getDatasetFileContentQueryKey(workspace, name, path),
      })
    );
  }
  return Promise.all(ops);
}
