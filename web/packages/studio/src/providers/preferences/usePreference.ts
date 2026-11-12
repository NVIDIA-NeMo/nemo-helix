// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getAuthGetMyPreferenceQueryKey,
  useAuthGetMyPreference,
} from '@nemo/sdk/generated/platform/profiles';
import type { PreferenceDocument } from '@nemo/sdk/generated/platform/schema';
import { featureFlags } from '@studio/constants/featureFlags';
import { useMe } from '@studio/providers/auth/useMe';
import { isJsonObject, mergePatch } from '@studio/providers/preferences/mergePatch';
import { readMirror, writeMirror } from '@studio/providers/preferences/mirror';
import type {
  JsonObject,
  PreferenceDefinition,
  PreferenceMapEntry,
  PreferenceScope,
} from '@studio/providers/preferences/types';
import {
  deferPreferencePatch,
  enqueuePreferencePatch,
  releaseDeferredPreferencePatches,
} from '@studio/providers/preferences/writeQueue';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo } from 'react';

/** [value, set, clear, ready]: `ready` is false until the stored value is known or its read has failed. */
type PreferenceState<T> = readonly [T, (value: T) => void, () => void, boolean];

const PREFERENCE_STALE_TIME_MS = 5 * 60_000;

const parse = <T>(definition: PreferenceDefinition<T>, raw: unknown): T => {
  if (raw === undefined) return definition.defaultValue;
  const result = definition.schema.safeParse(raw);
  return result.success ? result.data : definition.defaultValue;
};

const isMapEntry = (value: unknown): value is PreferenceMapEntry =>
  isJsonObject(value) && 'v' in value && typeof value.at === 'number';

const readRaw = (
  effective: JsonObject | undefined,
  definition: PreferenceDefinition<unknown>,
  entry?: string
): unknown => {
  const value = effective?.[definition.key];
  if (entry === undefined) return value;
  const stored = isJsonObject(value) ? value[entry] : undefined;
  return isMapEntry(stored) ? stored.v : undefined;
};

/** The merge patch that sets (or, with undefined, clears) one preference. */
const buildPatch = <T>(
  definition: PreferenceDefinition<T>,
  current: JsonObject | undefined,
  value: T | undefined,
  entry?: string
): JsonObject => {
  if (entry === undefined) {
    const next = value !== undefined && definition.cap ? definition.cap(value) : value;
    return { [definition.key]: next ?? null };
  }
  const map = isJsonObject(current?.[definition.key])
    ? (current[definition.key] as JsonObject)
    : {};
  const entries: JsonObject = {
    [entry]: value === undefined ? null : { v: value, at: Date.now() },
  };
  if (value !== undefined && definition.maxEntries) {
    const others = Object.entries(map)
      .filter(([key, stored]) => key !== entry && isMapEntry(stored))
      .sort(([, a], [, b]) => (b as PreferenceMapEntry).at - (a as PreferenceMapEntry).at);
    for (const [key] of others.slice(definition.maxEntries - 1)) entries[key] = null;
  }
  return { [definition.key]: entries };
};

const emptyDocument = (namespace: string, workspace?: string): PreferenceDocument => ({
  namespace,
  workspace,
  value: {},
  defaults: {},
  effective: {},
  db_version: 0,
});

const useServerPreference = <T>(
  definition: PreferenceDefinition<T>,
  { workspace, entry }: PreferenceScope = {}
): PreferenceState<T> => {
  const queryClient = useQueryClient();
  const me = useMe();
  const accountId = me.data?.account_id;
  const params = workspace ? { workspace } : undefined;
  const { data, isPlaceholderData, isError } = useAuthGetMyPreference(
    definition.namespace,
    params,
    {
      query: {
        enabled: Boolean(accountId),
        staleTime: PREFERENCE_STALE_TIME_MS,
        placeholderData: () =>
          accountId ? readMirror(accountId, definition.namespace, workspace) : undefined,
      },
    }
  );

  useEffect(() => {
    if (accountId) releaseDeferredPreferencePatches(accountId);
  }, [accountId]);

  useEffect(() => {
    if (accountId && data && !isPlaceholderData) writeMirror(accountId, data);
  }, [accountId, data, isPlaceholderData]);

  const value = useMemo(
    () =>
      parse(
        definition,
        readRaw(data?.effective, definition as PreferenceDefinition<unknown>, entry)
      ),
    [data?.effective, definition, entry]
  );

  const write = useCallback(
    (next: T | undefined) => {
      const queryKey = getAuthGetMyPreferenceQueryKey(definition.namespace, params);
      const current =
        queryClient.getQueryData<PreferenceDocument>(queryKey) ??
        emptyDocument(definition.namespace, workspace);
      const patch = buildPatch(definition, current.value, next, entry);
      const userValue = mergePatch(current.value, patch) as JsonObject;
      queryClient.setQueryData<PreferenceDocument>(queryKey, {
        ...current,
        value: userValue,
        effective: mergePatch(current.defaults ?? {}, userValue) as JsonObject,
      });
      // Before /me resolves the optimistic document stays fresh, so the GET won't revert it, and
      // the patch is sent once the account is known.
      const write = { namespace: definition.namespace, workspace, patch, queryClient };
      if (accountId) enqueuePreferencePatch({ ...write, accountId });
      else deferPreferencePatch(write);
    },
    // `params` is derived from `workspace`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [accountId, definition, entry, queryClient, workspace]
  );

  const setValue = useCallback((next: T) => write(next), [write]);
  const clearValue = useCallback(() => write(undefined), [write]);
  // Settled with an error (or no account to read for): the default is the best value we'll get.
  const ready = data !== undefined || isError || (!me.isPending && !accountId);
  return [value, setValue, clearValue, ready] as const;
};

const useLocalPreference = <T>(
  definition: PreferenceDefinition<T>,
  scope: PreferenceScope = {}
): PreferenceState<T> => {
  const [stored, setStored, clearStored] = useLocalStorage<unknown>(definition.legacyKey(scope));
  const value = useMemo(() => parse(definition, stored), [definition, stored]);
  const invalid = stored !== undefined && !definition.schema.safeParse(stored).success;
  // Drop values a schema no longer accepts (e.g. older formats) instead of leaving them to linger.
  useEffect(() => {
    if (invalid) clearStored();
  }, [clearStored, invalid]);
  const setValue = useCallback(
    (next: T) => setStored(definition.cap ? definition.cap(next) : next),
    [definition, setStored]
  );
  return [value, setValue, clearStored, true] as const;
};

/**
 * Reads and writes one Studio preference, like `useLocalStorage` but typed by its definition.
 *
 * With server preferences on, values roam with the signed-in account: reads come from the
 * namespace document (mirrored per account for first paint and offline use), and writes apply
 * optimistically and reach the server as one debounced merge patch per document. Otherwise they
 * stay in this browser's localStorage under the pre-server keys.
 */
export const usePreference: <T>(
  definition: PreferenceDefinition<T>,
  scope?: PreferenceScope
) => PreferenceState<T> = featureFlags.serverPreferencesEnabled
  ? useServerPreference
  : useLocalPreference;
