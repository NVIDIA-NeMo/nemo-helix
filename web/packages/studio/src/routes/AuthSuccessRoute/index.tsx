// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import {
  entitiesCreateWorkspace,
  entitiesGetWorkspace,
  getEntitiesGetWorkspaceQueryKey,
} from '@nemo/sdk/generated/platform/entity-store';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTES } from '@studio/constants/routes';
import { useAuthProfile } from '@studio/providers/auth';
import { getWorkspaceDetailsDefaultRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { FC, useCallback, useEffect, useRef } from 'react';
import { matchPath, useNavigate } from 'react-router';

type PersonalWorkspaceStatus = 'ready' | 'forbidden' | 'failed';

const isForbidden = (error: unknown) => isAxiosError(error) && error.response?.status === 403;

const checkAccess = (workspace: string): Promise<PersonalWorkspaceStatus> =>
  entitiesGetWorkspace(workspace).then(
    () => 'ready',
    (error) => {
      if (isForbidden(error)) return 'forbidden';
      logger.error(`Failed to look up workspace ${workspace}`, error);
      return 'failed';
    }
  );

/**
 * Creates the user's personal workspace if needed. `forbidden` is the expected outcome when
 * operators restrict WorkspaceCreator; `failed` is anything else.
 */
async function ensurePersonalWorkspace(
  workspace: string,
  name: string
): Promise<PersonalWorkspaceStatus> {
  try {
    await entitiesGetWorkspace(workspace);
    return 'ready';
  } catch (error) {
    if (!isAxiosError(error) || ![403, 404].includes(error.response?.status ?? 0)) {
      logger.error(`Failed to look up workspace ${workspace}`, error);
      return 'failed';
    }
  }

  try {
    await entitiesCreateWorkspace({
      name: workspace,
      description: `Automatically created workspace for ${name}`,
    });
    return 'ready';
  } catch (error) {
    // Either a concurrent login created it, or it exists but this user can't read it.
    if (isAxiosError(error) && error.response?.status === 409) {
      return checkAccess(workspace);
    }
    if (isForbidden(error)) return 'forbidden';
    logger.error(`Failed to create workspace ${workspace}`, error);
    return 'failed';
  }
}

const landsInWorkspace = (path: string, workspace: string) =>
  path === '/' ||
  !!matchPath('/workspaces', path) ||
  matchPath(`${ROUTES.workspace.index}/*`, path)?.params.workspace === workspace;

export const AuthSuccessRoute: FC = () => {
  const profile = useAuthProfile();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();

  const handleAuthenticated = useCallback(async () => {
    if (profile) {
      const { workspace, name } = profile;
      const status = await ensurePersonalWorkspace(workspace, name);
      if (status === 'failed') {
        toast.error(`Couldn't set up your personal workspace ${workspace}.`);
      }

      // WorkspaceProvider starts its access check before the workspace exists, so its 403 is
      // already cached (and not retried) by now. Refetch it before navigating, otherwise the
      // first login lands on the "no access to this workspace" screen until a manual reload.
      await queryClient.invalidateQueries({
        queryKey: getEntitiesGetWorkspaceQueryKey(workspace),
      });

      const path = profile.state?.path;
      if (status !== 'ready' && (!path || landsInWorkspace(path, workspace))) {
        navigate(getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE));
      } else if (path) {
        navigate({ pathname: path, search: profile.state?.search });
      } else {
        navigate('/');
      }
    }
  }, [navigate, profile, queryClient, toast]);

  // Workspace creation must run once per sign-in, not again whenever a dependency's identity changes.
  const hasHandledRef = useRef(false);
  useEffect(() => {
    if (!profile || hasHandledRef.current) return;
    hasHandledRef.current = true;
    handleAuthenticated();
  }, [handleAuthenticated, profile]);

  return <Loading />;
};
