// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { isSampleWorkspace } from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';

/**
 * Whether any visible workspace is a sample workspace; `undefined` until known.
 * Only the first page is fetched (shared with the workspace dropdown's cached query),
 * so a miss stays `undefined` when later pages could still hold a sample.
 */
export const useHasSampleWorkspace = (enabled = true): boolean | undefined => {
  const { data } = useEntitiesListWorkspaces(
    { page: 1, page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled, staleTime: 5_000 } }
  );
  if (!data?.data) {
    return undefined;
  }
  if (data.data.some((workspace) => isSampleWorkspace(workspace.name))) {
    return true;
  }
  const totalPages = data.pagination?.total_pages;
  return totalPages !== undefined && totalPages <= 1 ? false : undefined;
};
