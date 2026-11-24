// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getNextPageParam } from '@nemo/common/src/api/common/utils';
import {
  getSecretsListSecretsQueryKey,
  secretsListSecrets,
} from '@nemo/sdk/generated/platform/secrets';
import { useInfiniteQuery } from '@tanstack/react-query';

const SECRETS_PAGE_SIZE = 20;

/**
 * The pages of secrets in `workspace`. Every secret selector reads this one cache entry, so they
 * share a key and page shape; each decides for itself how many pages to load.
 */
export function useWorkspaceSecretPages(workspace: string, enabled: boolean) {
  return useInfiniteQuery({
    queryKey: [...getSecretsListSecretsQueryKey(workspace), 'infinite'] as const,
    queryFn: ({ signal, pageParam }) =>
      secretsListSecrets(workspace, { page: pageParam, page_size: SECRETS_PAGE_SIZE }, signal),
    initialPageParam: 1,
    getNextPageParam,
    enabled: enabled && !!workspace,
  });
}
