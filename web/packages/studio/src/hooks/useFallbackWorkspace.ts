// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useAuthProfile } from '@studio/providers/auth/useAuthProfile';
import { useMe } from '@studio/providers/auth/useMe';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { useMemo } from 'react';

const NO_EXCLUSIONS: readonly string[] = [];

/**
 * Where to send a user who has no usable workspace selected: the most recent workspace they can
 * still access, else their personal workspace, else the first workspace they can access, else
 * `default`. Never one in `exclude` (workspaces known to be forbidden), so a forbidden selection
 * can't fall back to itself or cycle between forbidden ones. Undefined while that can't be decided.
 */
export const useFallbackWorkspace = (
  exclude: readonly string[] = NO_EXCLUSIONS
): string | undefined => {
  const [recentWorkspaces, , , recentKnown] = usePreference(studioPreferences.recentWorkspaces);
  const personal = useAuthProfile()?.workspace || undefined;
  const me = useMe();
  const { data, isSuccess, isError } = useEntitiesListWorkspaces(
    { page: 1, page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: me.isSuccess || me.isError } }
  );

  return useMemo(() => {
    const usable = (name: string | undefined): name is string => !!name && !exclude.includes(name);
    const personalWorkspace = usable(personal) ? personal : undefined;
    if (!isSuccess && !isError) {
      // Optimistic until the list loads; the workspace access check still guards the choice.
      return recentKnown ? (recentWorkspaces.find(usable) ?? personalWorkspace) : undefined;
    }
    // Without the list, trust only names we know: excluding forbidden ones stops any cycling.
    const accessible = isError ? undefined : (data?.data ?? []).map(({ name }) => name);
    const reachable = (name: string | undefined) =>
      usable(name) && (!accessible || accessible.includes(name));
    return (
      recentWorkspaces.find(reachable) ??
      (reachable(personal) ? personal : undefined) ??
      accessible?.find(usable) ??
      (usable(DEFAULT_WORKSPACE) ? DEFAULT_WORKSPACE : undefined)
    );
  }, [data?.data, exclude, isError, isSuccess, personal, recentKnown, recentWorkspaces]);
};
