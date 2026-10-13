// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import {
  entitiesCreateWorkspace,
  entitiesGetWorkspace,
  getEntitiesGetWorkspaceQueryKey,
} from '@nemo/sdk/generated/platform/entity-store';
import { Loading } from '@studio/components/Layouts/Loading';
import { useAuthProfile } from '@studio/providers/auth';
import { useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { FC, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router';

async function createWorkspaceIfNotExists(workspace: string, name: string) {
  try {
    await entitiesGetWorkspace(workspace);
  } catch (error) {
    if (isAxiosError(error) && (error.response?.status === 404 || error.response?.status === 403)) {
      try {
        await entitiesCreateWorkspace({
          name: workspace,
          description: `Automatically created workspace for ${name}`,
        });
      } catch (createError) {
        // A concurrent login (another tab) may have created it first.
        if (!isAxiosError(createError) || createError.response?.status !== 409) {
          throw createError;
        }
      }
    } else {
      throw error;
    }
  }
}

export const AuthSuccessRoute: FC = () => {
  const profile = useAuthProfile();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();

  const handleAuthenticated = useCallback(async () => {
    if (profile) {
      const { workspace, name } = profile;
      try {
        await createWorkspaceIfNotExists(workspace, name);
      } catch (error) {
        toast.error(`Failed to create workspace ${workspace}: ${error}`);
        logger.error(`Failed to create workspace ${workspace}: ${error}`);
      }

      // WorkspaceProvider starts its access check before the workspace exists, so its 403 is
      // already cached (and not retried) by now. Refetch it before navigating, otherwise the
      // first login lands on the "no access to this workspace" screen until a manual reload.
      await queryClient.invalidateQueries({
        queryKey: getEntitiesGetWorkspaceQueryKey(workspace),
      });

      if (profile.state?.path) {
        navigate({
          pathname: profile.state.path,
          search: profile.state.search,
        });
      } else {
        navigate('/');
      }
    }
  }, [toast, navigate, profile, queryClient]);

  useEffect(() => {
    handleAuthenticated();
  }, [handleAuthenticated]);

  return <Loading />;
};
