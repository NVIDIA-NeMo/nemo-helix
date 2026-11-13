// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceSecretPages } from '@studio/routes/SecretsListRoute/useWorkspaceSecretPages';
import { useEffect } from 'react';

export interface WorkspaceSecrets {
  names: string[];
  /** The first page, or a later one the panel is still fetching, has not arrived. */
  isLoading: boolean;
  /** A page failed. `names` then holds only what loaded, which says nothing about what exists. */
  isError: boolean;
}

/**
 * Every secret name in `workspace`, for the panel's selectors. Pages are fetched until none
 * remain: a plain select has no scroll-to-load, and a missing name would read as an unselectable
 * secret.
 */
export function useWorkspaceSecrets(workspace: string, enabled: boolean): WorkspaceSecrets {
  const { data, isLoading, isError, hasNextPage, isFetchingNextPage, fetchNextPage } =
    useWorkspaceSecretPages(workspace, enabled);

  useEffect(() => {
    // After a failed page `hasNextPage` stays true, so without the `isError` guard this would
    // retry in a tight loop. `enabled` stops only automatic fetches, not this imperative call, so
    // a panel that closes mid-load must not keep paging.
    if (enabled && hasNextPage && !isFetchingNextPage && !isError) void fetchNextPage();
  }, [enabled, hasNextPage, isFetchingNextPage, isError, fetchNextPage]);

  return {
    names: data?.pages.flatMap((page) => page.data.map((secret) => secret.name)) ?? [],
    isLoading: !isError && (isLoading || hasNextPage),
    isError,
  };
}
