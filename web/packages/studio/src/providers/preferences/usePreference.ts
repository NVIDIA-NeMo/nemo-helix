// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAuthGetMyPreference } from '@nemo/sdk/generated/platform/profiles';
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
  enqueuePreferencePatch,
  getPreferenceOverlay,
  preferenceDocumentId,
  releasePreferencePatches,
  subscribePreferenceOverlays,
} from '@studio/providers/preferences/writeQueue';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useSyncExternalStore } from 'react';

/**
 * [value, set, clear, ready, loaded]: `ready` is false until the stored value is known or its read
 * has failed; `loaded` only once the server document itself has loaded. Read-modify-write updates
 * of whole values (arrays) should wait for `loaded`, or they replace what the server holds.
 */
type PreferenceState<T> = readonly [T, (value: T) => void, () => void, boolean, boolean];

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
    if (accountId) releasePreferencePatches(accountId);
  }, [accountId]);

  useEffect(() => {
    if (accountId && data && !isPlaceholderData) writeMirror(accountId, data);
  }, [accountId, data, isPlaceholderData]);

  // Edits not yet in the cached document, applied over whatever the server last returned, so a
  // GET that lands mid-edit (or before /me) can't revert them.
  const id = preferenceDocumentId(definition.namespace, workspace);
  const overlay = useSyncExternalStore(subscribePreferenceOverlays, () => getPreferenceOverlay(id));
  const userValue = useMemo(
    () => (overlay ? (mergePatch(data?.value ?? {}, overlay) as JsonObject) : data?.value),
    [data?.value, overlay]
  );
  const effective = useMemo(
    () => (overlay ? (mergePatch(data?.defaults ?? {}, userValue) as JsonObject) : data?.effective),
    [data?.defaults, data?.effective, overlay, userValue]
  );

  const value = useMemo(
    () => parse(definition, readRaw(effective, definition as PreferenceDefinition<unknown>, entry)),
    [effective, definition, entry]
  );

  const storedValue = useRef(data?.value);
  storedValue.current = data?.value;
  const write = useCallback(
    (next: T | undefined) => {
      // Read the overlay now, not from the render, so several writes in one tick compose.
      const pending = getPreferenceOverlay(id);
      const base = storedValue.current ?? {};
      const current = pending ? (mergePatch(base, pending) as JsonObject) : base;
      enqueuePreferencePatch({
        accountId,
        namespace: definition.namespace,
        workspace,
        patch: buildPatch(definition, current, next, entry),
        queryClient,
      });
    },
    [accountId, definition, entry, id, queryClient, workspace]
  );

  const setValue = useCallback((next: T) => write(next), [write]);
  const clearValue = useCallback(() => write(undefined), [write]);
  // Settled with an error (or no account to read for): the default is the best value we'll get.
  const ready = data !== undefined || isError || (!me.isPending && !accountId);
  const loaded = data !== undefined && !isPlaceholderData;
  return [value, setValue, clearValue, ready, loaded] as const;
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
  return [value, setValue, clearStored, true, true] as const;
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
