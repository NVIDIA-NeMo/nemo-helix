// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  authPatchMyPreference,
  getAuthGetMyPreferenceQueryKey,
} from '@nemo/sdk/generated/platform/profiles';
import type { PreferenceDocument, PreferenceValue } from '@nemo/sdk/generated/platform/schema';
import { composePatches } from '@studio/providers/preferences/mergePatch';
import { writeMirror } from '@studio/providers/preferences/mirror';
import type { JsonObject } from '@studio/providers/preferences/types';
import type { QueryClient } from '@tanstack/react-query';

export const PREFERENCE_WRITE_DEBOUNCE_MS = 500;

interface PendingWrite {
  patch: JsonObject;
  timer: ReturnType<typeof setTimeout>;
}

const pending = new Map<string, PendingWrite>();
// Latest request per document, so a slower earlier response can't overwrite a newer one.
const latestRequest = new Map<string, number>();

interface PreferenceWrite {
  readonly accountId: string;
  readonly namespace: string;
  readonly workspace?: string;
  readonly patch: JsonObject;
  readonly queryClient: QueryClient;
}

const flush = async (id: string, write: Omit<PreferenceWrite, 'patch'>) => {
  const entry = pending.get(id);
  if (!entry) return;
  pending.delete(id);
  const request = (latestRequest.get(id) ?? 0) + 1;
  latestRequest.set(id, request);
  const isLatest = () => latestRequest.get(id) === request && !pending.has(id);
  const params = write.workspace ? { workspace: write.workspace } : undefined;
  const queryKey = getAuthGetMyPreferenceQueryKey(write.namespace, params);
  try {
    const document = await authPatchMyPreference(
      write.namespace,
      entry.patch as PreferenceValue,
      params
    );
    // Newer edits are queued or in flight; their response will reconcile the cache.
    if (!isLatest()) return;
    writeMirror(write.accountId, document);
    write.queryClient.setQueryData<PreferenceDocument>(queryKey, document);
  } catch {
    if (isLatest()) await write.queryClient.invalidateQueries({ queryKey });
  }
};

/** Batches rapid edits to one namespace document into a single merge patch. */
export const enqueuePreferencePatch = ({ patch, ...write }: PreferenceWrite) => {
  const id = `${write.accountId}\u0000${write.namespace}\u0000${write.workspace ?? ''}`;
  const previous = pending.get(id);
  if (previous) clearTimeout(previous.timer);
  pending.set(id, {
    patch: previous ? composePatches(previous.patch, patch) : patch,
    timer: setTimeout(() => void flush(id, write), PREFERENCE_WRITE_DEBOUNCE_MS),
  });
};

type DeferredWrite = Omit<PreferenceWrite, 'accountId'>;

const deferred: DeferredWrite[] = [];

/** Holds a write made before the account is known; `releaseDeferredPreferencePatches` sends it. */
export const deferPreferencePatch = (write: DeferredWrite) => {
  deferred.push(write);
};

export const releaseDeferredPreferencePatches = (accountId: string) => {
  for (const write of deferred.splice(0)) enqueuePreferencePatch({ ...write, accountId });
};
