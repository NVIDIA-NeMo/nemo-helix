// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getSecretsListSecretsQueryKey,
  secretsListSecrets,
} from '@nemo/sdk/generated/platform/secrets';
import { useInfiniteQuery } from '@tanstack/react-query';
import { useEffect } from 'react';

const SECRETS_PAGE_SIZE = 20;

export interface WorkspaceSecrets {
  names: string[];
  isLoading: boolean;
}

/**
 * Every secret name in `workspace`, for the panel's selectors. Shares its query key and page shape
 * with `SecretSearchableSelect`, so both read one cache entry. Pages are fetched until none remain:
 * a plain select has no scroll-to-load, and a missing name would read as an unselectable secret.
 */
export function useWorkspaceSecrets(workspace: string, enabled: boolean): WorkspaceSecrets {
  const { data, isLoading, hasNextPage, isFetchingNextPage, fetchNextPage } = useInfiniteQuery({
    queryKey: [...getSecretsListSecretsQueryKey(workspace), 'infinite'] as const,
    queryFn: ({ signal, pageParam }) =>
      secretsListSecrets(workspace, { page: pageParam, page_size: SECRETS_PAGE_SIZE }, signal),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      const p = lastPage.pagination;
      return p && p.page < p.total_pages ? p.page + 1 : undefined;
    },
    enabled: enabled && !!workspace,
  });

  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage) void fetchNextPage();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  return {
    names: data?.pages.flatMap((page) => page.data.map((secret) => secret.name)) ?? [],
    isLoading: isLoading || hasNextPage,
  };
}
