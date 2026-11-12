// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { customFetch } from '@nemo/sdk/generated/fetchers/platform';
import { authGetMyPreference } from '@nemo/sdk/generated/platform/profiles';
import type { Theme } from '@nvidia/foundations-react-core';
import { featureFlags } from '@studio/constants/featureFlags';
import { useMe } from '@studio/providers/auth/useMe';
import { MIGRATED_PREFERENCES, studioPreferences } from '@studio/providers/preferences/definitions';
import { BROWSER_HAD_PREFERENCE_MIRROR_AT_BOOT } from '@studio/providers/preferences/mirror';
import { usePreference } from '@studio/providers/preferences/usePreference';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { UI_THEME_BOOT } from '@studio/util/localStorage';
import { useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { useEffect, useState } from 'react';

const PREFERENCES_QUERY_PREFIX = '/apis/auth/v2/me/preferences/';

const readLegacy = (key: string): unknown => {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? undefined : JSON.parse(raw);
  } catch {
    return undefined;
  }
};

/**
 * Seeds an account's empty documents from this browser's pre-server values. Only for the first
 * account to use server preferences here: the legacy keys don't say who wrote them.
 */
const migrateLegacyPreferences = async () => {
  for (const definition of MIGRATED_PREFERENCES) {
    const legacy = definition.schema.safeParse(readLegacy(definition.legacyKey({})));
    if (!legacy.success) continue;
    const document = await authGetMyPreference(definition.namespace);
    if (document.value[definition.key] !== undefined) continue;
    // Conditional, so a value set since the read (e.g. a theme picked right away) is never replaced.
    await customFetch({
      url: `${PREFERENCES_QUERY_PREFIX}${encodeURIComponent(definition.namespace)}`,
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', 'If-Match': String(document.db_version) },
      data: { [definition.key]: legacy.data },
    }).catch((error: unknown) => {
      if (!isAxiosError(error) || error.response?.status !== 412) throw error;
    });
  }
};

const usePreferenceMigration = (): boolean => {
  const accountId = useMe().data?.account_id;
  const queryClient = useQueryClient();
  const skip = !featureFlags.serverPreferencesEnabled || BROWSER_HAD_PREFERENCE_MIRROR_AT_BOOT;
  const [done, setDone] = useState(skip);

  useEffect(() => {
    if (skip || !accountId) return;
    migrateLegacyPreferences()
      .catch((error: unknown) => logger.error('Preference migration failed', error))
      .then(() =>
        queryClient.invalidateQueries({
          predicate: ({ queryKey }) =>
            typeof queryKey[0] === 'string' && queryKey[0].startsWith(PREFERENCES_QUERY_PREFIX),
        })
      )
      .finally(() => setDone(true));
  }, [accountId, queryClient, skip]);

  return done;
};

/** Keeps the boot theme (read before the account is known) in step with the account's theme. */
export const PreferencesSync = () => {
  const migrated = usePreferenceMigration();
  const [theme, , , ready] = usePreference(studioPreferences.theme);
  const [bootTheme, setBootTheme] = useLocalStorage<Theme>(UI_THEME_BOOT);

  useEffect(() => {
    if (migrated && ready && theme !== bootTheme) setBootTheme(theme);
  }, [bootTheme, migrated, ready, setBootTheme, theme]);

  return null;
};
