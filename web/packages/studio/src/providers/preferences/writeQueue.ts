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
import { notifyManager, type QueryClient } from '@tanstack/react-query';

export const PREFERENCE_WRITE_DEBOUNCE_MS = 500;

interface PreferenceWrite {
  readonly accountId?: string;
  readonly namespace: string;
  readonly workspace?: string;
  readonly patch: JsonObject;
  readonly queryClient: QueryClient;
}

interface DocumentWrites {
  /** Edits waiting for the debounce, or for the account to be known. */
  queued?: { patch: JsonObject; write: Omit<PreferenceWrite, 'patch'> };
  timer?: ReturnType<typeof setTimeout>;
  /** Sent but not yet answered, by request number. */
  inFlight: Map<number, JsonObject>;
  lastRequest: number;
  lastApplied: number;
  /** Everything not yet reflected in the cached document, composed in order. */
  overlay?: JsonObject;
}

const documents = new Map<string, DocumentWrites>();
const listeners = new Set<() => void>();

export const preferenceDocumentId = (namespace: string, workspace?: string) =>
  `${namespace}\u0000${workspace ?? ''}`;

const stateFor = (id: string): DocumentWrites => {
  let state = documents.get(id);
  if (!state) {
    state = { inFlight: new Map(), lastRequest: 0, lastApplied: 0 };
    documents.set(id, state);
  }
  return state;
};

const update = (state: DocumentWrites) => {
  const patches = [...state.inFlight.values()];
  if (state.queued) patches.push(state.queued.patch);
  state.overlay = patches.length ? patches.reduce(composePatches) : undefined;
  for (const listener of listeners) listener();
};

export const subscribePreferenceOverlays = (listener: () => void) => {
  listeners.add(listener);
  return () => void listeners.delete(listener);
};

/** Unsent and unanswered edits to one document, to apply over the cached document. */
export const getPreferenceOverlay = (id: string): JsonObject | undefined =>
  documents.get(id)?.overlay;

const flush = async (id: string) => {
  const state = stateFor(id);
  const queued = state.queued;
  if (!queued?.write.accountId) return;
  const { accountId, namespace, workspace, queryClient } = queued.write;
  state.queued = undefined;
  const request = ++state.lastRequest;
  state.inFlight.set(request, queued.patch);
  update(state);

  const params = workspace ? { workspace } : undefined;
  const queryKey = getAuthGetMyPreferenceQueryKey(namespace, params);
  try {
    const document = await authPatchMyPreference(
      namespace,
      queued.patch as PreferenceValue,
      params
    );
    // A slower earlier response must not overwrite a newer one.
    if (request > state.lastApplied) {
      state.lastApplied = request;
      writeMirror(accountId, document);
      queryClient.setQueryData<PreferenceDocument>(queryKey, document);
    }
  } catch {
    if (request === state.lastRequest) await queryClient.invalidateQueries({ queryKey });
  }
  // Drop the edit only after the cache notifies its observers, so the value never flickers back.
  notifyManager.schedule(() => {
    state.inFlight.delete(request);
    update(state);
  });
};

const schedule = (id: string, state: DocumentWrites) => {
  clearTimeout(state.timer);
  if (state.queued?.write.accountId) {
    state.timer = setTimeout(() => void flush(id), PREFERENCE_WRITE_DEBOUNCE_MS);
  }
};

/**
 * Batches rapid edits to one namespace document into a single merge patch. Without an account id
 * the edit is held (and still shown) until `releasePreferencePatches` supplies one.
 */
export const enqueuePreferencePatch = ({ patch, ...write }: PreferenceWrite) => {
  const id = preferenceDocumentId(write.namespace, write.workspace);
  const state = stateFor(id);
  state.queued = {
    patch: state.queued ? composePatches(state.queued.patch, patch) : patch,
    write: { ...write, accountId: write.accountId ?? state.queued?.write.accountId },
  };
  schedule(id, state);
  update(state);
};

export const releasePreferencePatches = (accountId: string) => {
  for (const [id, state] of documents) {
    if (!state.queued || state.queued.write.accountId) continue;
    state.queued = { ...state.queued, write: { ...state.queued.write, accountId } };
    schedule(id, state);
  }
};
