// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { WORKSPACE_DROPDOWN_RECENT_LIMIT } from '@studio/components/WorkspaceDropdown/constants';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { studioPreferences } from '@studio/providers/preferences/definitions';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { useEffect, useRef } from 'react';

export const useRecentWorkspaces = () => {
  // The preference schema guarantees an array, even if the stored value is invalid.
  const [recentWorkspaces, setRecentWorkspaces, , , loaded] = usePreference(
    studioPreferences.recentWorkspaces
  );
  const workspace = useWorkspaceFromPathIfExists();

  const addRecentWorkspace = (name: string) => {
    const existingIndex = recentWorkspaces.findIndex((w) => w === name);

    if (existingIndex !== -1) {
      // Workspace exists, move to front
      setRecentWorkspaces([name, ...recentWorkspaces.filter((w) => w !== name)]);
    } else if (recentWorkspaces.length >= WORKSPACE_DROPDOWN_RECENT_LIMIT) {
      // Add new workspace and remove oldest
      setRecentWorkspaces([
        name,
        ...recentWorkspaces.slice(0, WORKSPACE_DROPDOWN_RECENT_LIMIT - 1),
      ]);
    } else {
      // Add new workspace
      setRecentWorkspaces([name, ...recentWorkspaces]);
    }
  };

  // We should always use the current workspace in the recent workspaces list. Wait for the stored
  // list: the write replaces the whole array, so writing over the default would wipe it.
  const added = useRef(false);
  useEffect(() => {
    if (!loaded || added.current) return;
    added.current = true;
    if (workspace && recentWorkspaces[0] !== workspace) {
      addRecentWorkspace(workspace);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded]);

  return {
    recentWorkspaces,
    setRecentWorkspaces,
    addRecentWorkspace,
    removeRecentWorkspace: (name: string) => {
      setRecentWorkspaces(recentWorkspaces.filter((w) => w !== name));
    },
  };
};
