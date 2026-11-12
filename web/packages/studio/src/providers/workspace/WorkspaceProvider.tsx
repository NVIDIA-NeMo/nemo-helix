// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesGetWorkspace } from '@nemo/sdk/generated/platform/entity-store';
import { useFallbackWorkspace } from '@studio/hooks/useFallbackWorkspace';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { WorkspaceContext } from '@studio/providers/workspace/WorkspaceContext';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { SELECTED_WORKSPACE_KEY } from '@studio/util/localStorage';
import { type FC, type ReactNode, useCallback, useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router';

interface WorkspaceProviderProps {
  children: ReactNode;
}

export const WorkspaceProvider: FC<WorkspaceProviderProps> = ({ children }) => {
  const { isTokenActive } = useAuthTokenStatus();
  // Per browser, not roaming: devices on different workspaces must not overwrite each other.
  const [storedWorkspace, setStoredWorkspace] = useLocalStorage<string>(SELECTED_WORKSPACE_KEY);
  const { workspace: workspaceParam } = useParams();
  // Workspaces that returned 403 in this session, so auto-fallback can't cycle between them.
  const [forbidden, setForbidden] = useState<readonly string[]>([]);
  const excluded = useMemo(
    () => (storedWorkspace ? [...forbidden, storedWorkspace] : forbidden),
    [forbidden, storedWorkspace]
  );
  const fallbackWorkspace = useFallbackWorkspace(excluded);

  // Priority: URL route param → stored selection → fallback.
  const selectedWorkspace = workspaceParam || storedWorkspace || fallbackWorkspace;

  // Persist route changes, but do not react to storedWorkspace changes. Otherwise, tabs on
  // different workspace routes continuously overwrite each other's stored value.
  useEffect(() => {
    if (workspaceParam) {
      setStoredWorkspace(workspaceParam);
    }
  }, [workspaceParam, setStoredWorkspace]);

  const { isPending: isWorkspaceLoading, error: workspaceError } = useEntitiesGetWorkspace(
    selectedWorkspace ?? '',
    {
      query: {
        enabled: !!selectedWorkspace && isTokenActive,
        retry: (failureCount, err) => {
          const status = err.response?.status;
          if (status === 401 || status === 403) return false;
          return failureCount < 3;
        },
      },
    }
  );

  const hasUnauthorizedError = !isWorkspaceLoading && workspaceError?.response?.status === 403;

  // Auto-fall-back when a forbidden stored workspace is used implicitly (no URL param).
  const willAutoFallback =
    hasUnauthorizedError && !workspaceParam && !!storedWorkspace && !!fallbackWorkspace;

  useEffect(() => {
    if (willAutoFallback && storedWorkspace && fallbackWorkspace) {
      setForbidden((previous) => [...previous, storedWorkspace]);
      setStoredWorkspace(fallbackWorkspace);
    }
  }, [willAutoFallback, storedWorkspace, fallbackWorkspace, setStoredWorkspace]);

  // Suppress during fallback to avoid a flash of the unauthorized screen.
  const isWorkspaceUnauthorized = hasUnauthorizedError && !willAutoFallback;

  const setSelectedWorkspace = useCallback(
    (workspace: string) => {
      setStoredWorkspace(workspace);
    },
    [setStoredWorkspace]
  );

  const value = useMemo(
    () => ({
      selectedWorkspace,
      setSelectedWorkspace,
      isWorkspaceUnauthorized,
      isWorkspaceLoading,
    }),
    [selectedWorkspace, setSelectedWorkspace, isWorkspaceUnauthorized, isWorkspaceLoading]
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
};
