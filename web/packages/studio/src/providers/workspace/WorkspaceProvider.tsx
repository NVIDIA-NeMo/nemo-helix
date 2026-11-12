// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEntitiesGetWorkspace } from '@nemo/sdk/generated/platform/entity-store';
import { useFallbackWorkspace } from '@studio/hooks/useFallbackWorkspace';
import { useAuthTokenStatus } from '@studio/providers/auth/useAuthTokenStatus';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { WorkspaceContext } from '@studio/providers/workspace/WorkspaceContext';
import { type FC, type ReactNode, useCallback, useEffect, useMemo } from 'react';
import { useParams } from 'react-router';

interface WorkspaceProviderProps {
  children: ReactNode;
}

export const WorkspaceProvider: FC<WorkspaceProviderProps> = ({ children }) => {
  const { isTokenActive } = useAuthTokenStatus();
  const [storedWorkspace, setStoredWorkspace, , storedKnown] = usePreference(
    studioPreferences.selectedWorkspace
  );
  const { workspace: workspaceParam } = useParams();
  // Never the stored workspace, so it is also a valid destination when that one is forbidden.
  const fallbackWorkspace = useFallbackWorkspace(storedWorkspace);

  // Priority: URL route param → stored selection → fallback. Wait for the stored selection rather
  // than briefly selecting the fallback.
  const selectedWorkspace = storedKnown
    ? workspaceParam || storedWorkspace || fallbackWorkspace
    : workspaceParam;

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
    if (willAutoFallback && fallbackWorkspace) {
      setStoredWorkspace(fallbackWorkspace);
    }
  }, [willAutoFallback, fallbackWorkspace, setStoredWorkspace]);

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
