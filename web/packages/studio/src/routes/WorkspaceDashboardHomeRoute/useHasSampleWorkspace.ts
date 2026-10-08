// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { isSampleWorkspace } from '@studio/routes/WorkspaceDashboardHomeRoute/useSampleQuickstartAgent';

/**
 * Whether any visible workspace is a sample workspace; `undefined` until known.
 * Shares the workspace dropdown's cached query.
 */
export const useHasSampleWorkspace = (enabled = true): boolean | undefined => {
  const { data } = useEntitiesListWorkspaces(
    { page: 1, page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled, staleTime: 5_000 } }
  );
  return data?.data?.some((workspace) => isSampleWorkspace(workspace.name));
};
