// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Theme } from '@nvidia/foundations-react-core';
import {
  type TrendVisibilityChoice,
  trendVisibilityStorageKey,
} from '@studio/components/charts/ExperimentTrendChart/visibility';
import { WORKSPACE_DROPDOWN_RECENT_LIMIT } from '@studio/components/WorkspaceDropdown/constants';
import type { PreferenceDefinition } from '@studio/providers/preferences/types';
import { getQuickstartDismissedKey } from '@studio/routes/WorkspaceDashboardHomeRoute/quickstartDismissedStorage';
import { TOUR_SEEN_KEY, UI_THEME, WORKSPACE_DROPDOWN_RECENT_KEY } from '@studio/util/localStorage';
import { z } from 'zod';

/**
 * Studio's roaming preferences. The server stores them as opaque JSON, so these schemas are the
 * contract: invalid stored values fall back to the default rather than reaching the UI.
 */
export const studioPreferences = {
  theme: {
    namespace: 'studio.ui',
    key: 'theme',
    schema: z.enum(['light', 'dark', 'system']) as z.ZodType<Theme>,
    defaultValue: 'dark',
    legacyKey: () => UI_THEME,
  } satisfies PreferenceDefinition<Theme>,
  recentWorkspaces: {
    namespace: 'studio.ui',
    key: 'recentWorkspaces',
    schema: z.array(z.string()),
    defaultValue: [],
    legacyKey: () => WORKSPACE_DROPDOWN_RECENT_KEY,
    cap: (value: string[]) => value.slice(0, WORKSPACE_DROPDOWN_RECENT_LIMIT),
  } satisfies PreferenceDefinition<string[]>,
  tourSeen: {
    namespace: 'studio.onboarding',
    key: 'tourSeen',
    schema: z.boolean(),
    defaultValue: false,
    legacyKey: () => TOUR_SEEN_KEY,
  } satisfies PreferenceDefinition<boolean>,
  quickstartDismissed: {
    namespace: 'studio.onboarding',
    key: 'quickstartDismissed',
    schema: z.boolean(),
    defaultValue: false,
    legacyKey: ({ workspace }) => getQuickstartDismissedKey(workspace ?? ''),
  } satisfies PreferenceDefinition<boolean>,
  experimentTrendVisibility: {
    namespace: 'studio.experiments',
    key: 'trendVisibility',
    schema: z.object({ visible: z.boolean(), flag: z.boolean() }).optional(),
    defaultValue: undefined,
    legacyKey: ({ entry }) => trendVisibilityStorageKey(entry),
    maxEntries: 50,
  } satisfies PreferenceDefinition<TrendVisibilityChoice | undefined>,
};

/** Legacy values that reveal nothing about whoever wrote them, so they may seed a new account. */
export const MIGRATED_PREFERENCES: readonly PreferenceDefinition<unknown>[] = [
  studioPreferences.theme,
  studioPreferences.tourSeen,
];
