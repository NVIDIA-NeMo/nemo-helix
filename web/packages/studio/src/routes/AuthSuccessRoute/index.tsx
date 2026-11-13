// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
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
import { FC, useCallback, useEffect } from 'react';
import { matchPath, useNavigate } from 'react-router';

const canReadWorkspace = (workspace: string) =>
  entitiesGetWorkspace(workspace).then(
    () => true,
    () => false
  );

/** Creates the user's personal workspace if needed and reports whether they can open it. */
async function ensurePersonalWorkspace(workspace: string, name: string): Promise<boolean> {
  try {
    await entitiesGetWorkspace(workspace);
    return true;
  } catch (error) {
    if (!isAxiosError(error) || ![403, 404].includes(error.response?.status ?? 0)) {
      logger.warn(`Failed to look up workspace ${workspace}`, error);
      return false;
    }
  }

  try {
    await entitiesCreateWorkspace({
      name: workspace,
      description: `Automatically created workspace for ${name}`,
    });
    return true;
  } catch (error) {
    // Either a concurrent login created it, or it exists but this user can't read it.
    if (isAxiosError(error) && error.response?.status === 409) {
      return canReadWorkspace(workspace);
    }
    // 403 is expected when operators restrict WorkspaceCreator.
    logger.warn(`Could not create workspace ${workspace}`, error);
    return false;
  }
}

const landsInWorkspace = (path: string, workspace: string) =>
  path === '/' ||
  !!matchPath('/workspaces', path) ||
  matchPath(`${ROUTES.workspace.index}/*`, path)?.params.workspace === workspace;

export const AuthSuccessRoute: FC = () => {
  const profile = useAuthProfile();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const handleAuthenticated = useCallback(async () => {
    if (profile) {
      const { workspace, name } = profile;
      const hasPersonalWorkspace = await ensurePersonalWorkspace(workspace, name);

      // WorkspaceProvider starts its access check before the workspace exists, so its 403 is
      // already cached (and not retried) by now. Refetch it before navigating, otherwise the
      // first login lands on the "no access to this workspace" screen until a manual reload.
      await queryClient.invalidateQueries({
        queryKey: getEntitiesGetWorkspaceQueryKey(workspace),
      });

      const path = profile.state?.path;
      if (!hasPersonalWorkspace && (!path || landsInWorkspace(path, workspace))) {
        navigate(getWorkspaceDetailsDefaultRoute(DEFAULT_WORKSPACE));
      } else if (path) {
        navigate({ pathname: path, search: profile.state?.search });
      } else {
        navigate('/');
      }
    }
  }, [navigate, profile, queryClient]);

  useEffect(() => {
    handleAuthenticated();
  }, [handleAuthenticated]);

  return <Loading />;
};
