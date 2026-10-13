/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { isForbiddenError } from '@nemo/common/src/api/common/utils';
import { useEntitiesListWorkspaceMembers } from '@nemo/sdk/generated/platform/entity-store';
import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';

/**
 * Lists the workspace's members, with the permission failure surfaced as `isForbidden`.
 *
 * A 403 means the caller simply isn't allowed to read membership — retrying can't change that, so
 * we stop immediately instead of burning the default three retries before showing an error screen.
 * Every consumer of this list must go through this hook so they agree on the retry behavior.
 */
export const useWorkspaceMembers = (workspace: string) => {
  const query = useEntitiesListWorkspaceMembers(workspace, {
    query: {
      retry: (failureCount, error) =>
        !isForbiddenError(error) && failureCount < DEFAULT_QUERY_RETRY_COUNT,
    },
  });

  return { ...query, isForbidden: isForbiddenError(query.error) };
};
