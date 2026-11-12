// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PreferenceDocument } from '@nemo/sdk/generated/platform/schema';

/**
 * Last known server documents, keyed by account so a shared browser never shows one user's
 * preferences to another. Used for first render and when the server is unreachable.
 */
export const MIRROR_PREFIX = 'nemo-studio:prefs:';

const mirrorKey = (accountId: string, namespace: string, workspace?: string) =>
  `${MIRROR_PREFIX}${accountId}:${namespace}:${workspace ?? ''}`;

export const readMirror = (
  accountId: string,
  namespace: string,
  workspace?: string
): PreferenceDocument | undefined => {
  try {
    const raw = window.localStorage.getItem(mirrorKey(accountId, namespace, workspace));
    return raw ? (JSON.parse(raw) as PreferenceDocument) : undefined;
  } catch {
    return undefined;
  }
};

export const writeMirror = (accountId: string, document: PreferenceDocument) => {
  try {
    window.localStorage.setItem(
      mirrorKey(accountId, document.namespace, document.workspace),
      JSON.stringify(document)
    );
  } catch {
    // Storage full or blocked; the server copy is authoritative anyway.
  }
};

const hasAnyMirror = () => {
  try {
    return Object.keys(window.localStorage).some((key) => key.startsWith(MIRROR_PREFIX));
  } catch {
    return true;
  }
};

/**
 * Captured before any mirror is written this page load: legacy keys may only be migrated into the
 * first account to use server preferences in this browser, since they don't record who wrote them.
 */
export const BROWSER_HAD_PREFERENCE_MIRROR_AT_BOOT =
  typeof window === 'undefined' ? true : hasAnyMirror();
