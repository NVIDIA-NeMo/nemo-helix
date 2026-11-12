// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { useEntitiesListWorkspaces } from '@nemo/sdk/generated/platform/entity-store';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useMe } from '@studio/providers/auth/useMe';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { useMemo } from 'react';

/**
 * Where to send a user who has no usable workspace selected: the most recent workspace they can
 * still access, else the first workspace they can access, else `default`. Never `exclude` (the
 * workspace that just failed), so a forbidden selection can't fall back to itself. Undefined while
 * that can't be decided yet.
 */
export const useFallbackWorkspace = (exclude?: string): string | undefined => {
  const [recentWorkspaces, , , recentKnown] = usePreference(studioPreferences.recentWorkspaces);
  const me = useMe();
  const { data, isSuccess, isError } = useEntitiesListWorkspaces(
    { page: 1, page_size: DEFAULT_LARGE_PAGE_SIZE },
    { query: { enabled: me.isSuccess || me.isError } }
  );

  return useMemo(() => {
    const usable = (name: string) => name !== exclude;
    if (!isSuccess && !isError) {
      // Optimistic until the list loads; the workspace access check still guards the choice.
      return recentKnown ? recentWorkspaces.find(usable) : undefined;
    }
    const accessible = (data?.data ?? []).map(({ name }) => name);
    return (
      recentWorkspaces.find((name) => usable(name) && (isError || accessible.includes(name))) ??
      accessible.find(usable) ??
      (usable(DEFAULT_WORKSPACE) ? DEFAULT_WORKSPACE : undefined)
    );
  }, [data?.data, exclude, isError, isSuccess, recentKnown, recentWorkspaces]);
};
