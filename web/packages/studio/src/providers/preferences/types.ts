// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { z } from 'zod';

export type JsonObject = Record<string, unknown>;

/** Where one preference lives within a namespace document. */
export interface PreferenceScope {
  /** Workspace-scoped preferences are stored per workspace; omit for global. */
  readonly workspace?: string;
  /** For map preferences: the entry within the map (e.g. an experiment group id). */
  readonly entry?: string;
}

export interface PreferenceDefinition<T> {
  readonly namespace: string;
  readonly key: string;
  readonly schema: z.ZodType<T>;
  readonly defaultValue: T;
  /** The pre-server localStorage key, used while server preferences are off and for migration. */
  readonly legacyKey: (scope: PreferenceScope) => string;
  /** Map preferences: keep only this many most recently written entries. */
  readonly maxEntries?: number;
  /** Bound values the client grows (e.g. recent lists) before they are written. */
  readonly cap?: (value: T) => T;
}

/** Map preferences store each entry with its write time so the oldest can be evicted. */
export interface PreferenceMapEntry {
  readonly v: unknown;
  readonly at: number;
}

export type PreferenceSetter<T> = (value: T | undefined) => void;
