// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesGetWorkspace } from '@nemo/sdk/generated/platform/entity-store';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { WorkspaceContext } from '@studio/providers/workspace/WorkspaceContext';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { SELECTED_WORKSPACE_KEY } from '@studio/util/localStorage';
import { FC, ReactNode, useCallback, useEffect, useMemo } from 'react';
import { useParams } from 'react-router';

interface WorkspaceProviderProps {
  children: ReactNode;
}

export const WorkspaceProvider: FC<WorkspaceProviderProps> = ({ children }) => {
  const profile = useAuthProfile();
  const { isTokenActive } = useAuthTokenStatus();
  const [storedWorkspace, setStoredWorkspace] = useLocalStorage<string>(SELECTED_WORKSPACE_KEY);
  const { workspace: workspaceParam } = useParams();

  // Priority: URL route param → localStorage → profile namespace (default)
  const selectedWorkspace = workspaceParam || storedWorkspace || profile?.workspace;

  // Sync localStorage with profile namespace on first load if nothing is stored
  useEffect(() => {
    if (!storedWorkspace && profile?.workspace) {
      setStoredWorkspace(profile.workspace);
    }
  }, [profile?.workspace, storedWorkspace, setStoredWorkspace]);

  // Persist route changes, but do not react to storedWorkspace changes. Otherwise, tabs on
  // different workspace routes continuously overwrite each other's shared localStorage value.
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
    hasUnauthorizedError &&
    !workspaceParam &&
    !!storedWorkspace &&
    !!profile?.workspace &&
    storedWorkspace !== profile.workspace;

  useEffect(() => {
    if (willAutoFallback && profile?.workspace) {
      setStoredWorkspace(profile.workspace);
    }
  }, [willAutoFallback, profile?.workspace, setStoredWorkspace]);

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
