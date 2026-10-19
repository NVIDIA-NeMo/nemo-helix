/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * Local feature-flag overrides.
 *
 * Adds a localStorage layer on top of the normal env cascade so a developer can
 * flip flags without a rebuild or a redeploy. Overrides are read once at module
 * load, so a page reload is required for a change to take effect — that is
 * intentional: the parsed `featureFlags` object stays a plain immutable value
 * rather than becoming reactive state that all ~30 derived constants in
 * `environment.ts` would have to subscribe to.
 *
 * Storage is a single key, `feature-flag-overrides`, holding
 * `{ "<camelCaseFlagKey>": "<raw string value>" }`. Raw strings are stored so
 * the existing Zod schemas in `utils.ts` do all validation and coercion; there
 * is no second parsing path to keep in sync.
 *
 * This module owns storage only. The editing UI lives in
 * `components/FeatureFlagPanel` and composes these functions with
 * `flagDefinitions` directly.
 */

import { logger } from '@nemo/common/src/utils/logger';
import { EnvConfig, FlagDescriptor } from '@studio/constants/featureFlags/utils';
import { FEATURE_FLAG_OVERRIDES_KEY } from '@studio/util/localStorage';

/** Raw string values keyed by flag name, exactly as stored. */
export type OverrideBlob = Record<string, string>;

/**
 * Flags an override must never be able to change.
 *
 * The gate decides whether overrides apply at all, so it has to come from the
 * build environment alone. Two different failures follow from letting it be
 * overridden, and the second is the easy one to miss:
 *
 * - `false` -> `true` is harmless. Where the deployment ships the gate off, the
 *   override layer is never consulted, so the entry is inert.
 * - `true` -> `false` strands the user. The cascade decision is made from the
 *   build value, so overrides keep applying, but `environment.ts` derives
 *   `FLAG_OVERRIDES_ENABLED` from the *parsed* value and hides the editor. The
 *   one control that could undo it is gone, leaving only hand-editing storage.
 *
 * Filtered here rather than at either call site so the parse cascade and the
 * panel cannot drift apart on it.
 */
export const NON_OVERRIDABLE_FLAGS: ReadonlySet<string> = new Set(['flagOverridesEnabled']);

/**
 * Reach localStorage without assuming it exists or is accessible.
 *
 * Reading `window.localStorage` is itself a throwing operation when storage is
 * blocked — third-party storage disabled, or a sandboxed iframe — so a `typeof`
 * guard is not safe: it evaluates the getter and throws with it. Every access in
 * this module funnels through here so a blocked-storage browser degrades to "no
 * overrides" instead of taking down app boot.
 */
const safeStorage = (): Storage | null => {
  try {
    if (typeof window === 'undefined') return null;
    return window.localStorage ?? null;
  } catch {
    return null;
  }
};

/**
 * Read the raw overrides blob.
 *
 * Any malformed input is discarded wholesale rather than partially applied — a
 * dev convenience must never be able to white-screen the app.
 */
const readOverridesBlob = (): Record<string, unknown> => {
  const storage = safeStorage();
  if (!storage) return {};

  let raw: string | null = null;
  try {
    raw = storage.getItem(FEATURE_FLAG_OVERRIDES_KEY);
  } catch {
    // getItem can still throw even once the getter itself succeeded.
    return {};
  }
  if (!raw) return {};

  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      logger.warn(
        `[featureFlags] Ignoring "${FEATURE_FLAG_OVERRIDES_KEY}": expected a JSON object, got ${Array.isArray(parsed) ? 'an array' : typeof parsed}.`
      );
      return {};
    }
    return parsed as Record<string, unknown>;
  } catch {
    logger.warn(`[featureFlags] Ignoring "${FEATURE_FLAG_OVERRIDES_KEY}": not valid JSON.`);
    return {};
  }
};

/**
 * Stored entries that still correspond to a real flag and a value that flag
 * accepts.
 *
 * The blob outlives the code that produced it: flags get deleted and change
 * type while overrides persist in a developer's browser indefinitely, so most
 * of what this drops is *stale*, not hostile. Dropping has to happen before the
 * values reach `parseFlags`, which treats a schema failure as a fatal "missing
 * required flag" and would take the app down over a leftover entry.
 */
const validEntries = (
  definitions: Record<string, FlagDescriptor>
): Array<{ key: string; descriptor: FlagDescriptor; value: string }> => {
  const entries: Array<{ key: string; descriptor: FlagDescriptor; value: string }> = [];

  for (const [key, value] of Object.entries(readOverridesBlob())) {
    // `Object.hasOwn`, not a truthiness check: a bare `definitions[key]` lookup
    // reaches Object.prototype, so a stored key like "constructor" or "toString"
    // returns a truthy non-descriptor whose `.schema` is undefined. Calling
    // `.safeParse` on that throws a TypeError out of this warn-and-skip loop and
    // aborts flag parsing entirely.
    if (!Object.hasOwn(definitions, key)) {
      logger.warn(`[featureFlags] Ignoring override for unknown flag "${key}".`);
      continue;
    }
    if (NON_OVERRIDABLE_FLAGS.has(key)) {
      logger.warn(`[featureFlags] Ignoring override for "${key}": this flag cannot be overridden.`);
      continue;
    }
    const descriptor = definitions[key]!;
    if (typeof value !== 'string') {
      logger.warn(
        `[featureFlags] Ignoring override for "${key}": expected a string, got ${typeof value}.`
      );
      continue;
    }
    if (!descriptor.schema.safeParse(value).success) {
      logger.warn(
        `[featureFlags] Ignoring override for "${key}": "${value}" is not a valid ${descriptor.typeName}.`
      );
      continue;
    }
    entries.push({ key, descriptor, value });
  }

  return entries;
};

/**
 * Build the env-shaped override layer to merge over `import.meta.env`.
 *
 * Returns env-var-keyed strings so the caller can spread it straight into the
 * `EnvConfig` handed to `parseFlags`.
 */
export const readFlagOverrides = (definitions: Record<string, FlagDescriptor>): EnvConfig => {
  const layer: EnvConfig = {};
  for (const { descriptor, value } of validEntries(definitions)) {
    layer[descriptor.envVar] = value;
  }
  return layer;
};

/**
 * The same entries keyed by flag name, for the editing UI.
 *
 * Deliberately filtered rather than raw: the panel should show what the app
 * would actually apply on reload, not what happens to be sitting in storage.
 */
export const readStoredOverrides = (definitions: Record<string, FlagDescriptor>): OverrideBlob => {
  const blob: OverrideBlob = {};
  for (const { key, value } of validEntries(definitions)) {
    blob[key] = value;
  }
  return blob;
};

/**
 * Persist the overrides blob. Returns false when storage rejected the write
 * (quota exceeded, storage blocked) so callers can report it rather than
 * claiming a save that did not happen.
 *
 * An empty blob removes the key outright instead of storing `"{}"`, so a
 * cleared state is indistinguishable from one that was never set.
 */
export const writeStoredOverrides = (blob: OverrideBlob): boolean => {
  const storage = safeStorage();
  if (!storage) return false;

  try {
    if (Object.keys(blob).length === 0) {
      storage.removeItem(FEATURE_FLAG_OVERRIDES_KEY);
    } else {
      storage.setItem(FEATURE_FLAG_OVERRIDES_KEY, JSON.stringify(blob));
    }
    return true;
  } catch (error) {
    logger.error('[featureFlags] Failed to save overrides.', error);
    return false;
  }
};
