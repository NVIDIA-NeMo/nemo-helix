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
 * Adds a localStorage layer on top of the normal env cascade so a developer (or
 * the companion Chrome extension) can flip flags without a rebuild or a
 * redeploy. Overrides are read once at module load, so a page reload is
 * required for a change to take effect — that is intentional: the parsed
 * `featureFlags` object stays a plain immutable value rather than becoming
 * reactive state every consumer would have to subscribe to.
 *
 * Two localStorage keys form the contract:
 *
 * - `feature-flag-overrides` — written by the user/extension, read by the app.
 *   `{ "<camelCaseFlagKey>": "<raw string value>" }`. Raw strings are stored so
 *   the existing Zod schemas in `utils.ts` do all validation and coercion;
 *   there is no second parsing path to keep in sync.
 *
 * - `feature-flag-manifest` — written by the app, read by the extension.
 *   Describes every known flag, its type, and its effective value. A Chrome
 *   content script runs in an isolated world and cannot see page-world globals
 *   like `window.__flags`, but it does share the page's localStorage — so the
 *   manifest, not the global, is the extension's interface.
 */

import { logger } from '@nemo/common/src/utils/logger';
import { EnvConfig, FlagDescriptor } from '@studio/constants/featureFlags/utils';
import { FEATURE_FLAG_MANIFEST_KEY, FEATURE_FLAG_OVERRIDES_KEY } from '@studio/util/localStorage';

/** Bump when the manifest shape changes incompatibly, so the extension can refuse to render. */
export const FLAG_MANIFEST_VERSION = 1;

export type FlagManifestEntry = {
  /** camelCase key used in app code and in the overrides blob. */
  key: string;
  /** Underlying `VITE_FF_*` environment variable. */
  envVar: string;
  /** Drives which control the extension renders: toggle, tri-state, or free text. */
  type: string;
  /** Value with no override applied — what "reset" returns to. */
  baseValue: unknown;
  /** Value the app is actually running with. */
  effectiveValue: unknown;
  /**
   * Whether an override for this flag is present in storage.
   *
   * Not the same as "applied": between a write and the next reload an override
   * is stored but the page is still running `effectiveValue`. Both fields are
   * accurate in that window — `overridden` describes storage, `effectiveValue`
   * describes the running app — so a reader wanting to distinguish staged from
   * applied compares the two rather than trusting either alone.
   */
  overridden: boolean;
};

export type FlagManifest = {
  version: number;
  /** Mirrors the server-controlled gate so the extension can explain itself when disabled. */
  overridesEnabled: boolean;
  /** Told to the extension rather than hardcoded there, so the key can move. */
  overridesKey: string;
  flags: FlagManifestEntry[];
};

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
 * Build the env-shaped override layer to merge over `import.meta.env`.
 *
 * Returns env-var-keyed strings so the caller can spread it directly into the
 * `EnvConfig` handed to `parseFlags`. Entries that name an unknown flag, or
 * whose value fails its flag's schema, are dropped with a warning: `parseFlags`
 * treats a schema failure as a fatal "missing required flag", so letting a bad
 * override reach it would take down the whole app.
 */
export const readFlagOverrides = (definitions: Record<string, FlagDescriptor>): EnvConfig => {
  const blob = readOverridesBlob();
  const layer: EnvConfig = {};

  for (const [key, value] of Object.entries(blob)) {
    // `Object.hasOwn`, not a truthiness check: a bare `definitions[key]` lookup
    // reaches Object.prototype, so a stored key like "constructor" or "toString"
    // returns a truthy non-descriptor whose `.schema` is undefined. Calling
    // `.safeParse` on that throws a TypeError out of this per-key warn-and-skip
    // loop and aborts flag parsing entirely — a white screen from a typo.
    if (!Object.hasOwn(definitions, key)) {
      logger.warn(`[featureFlags] Ignoring override for unknown flag "${key}".`);
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
    layer[descriptor.envVar] = value;
  }

  return layer;
};

/** Flag keys that currently have an override applied, for manifest annotation. */
const overriddenKeys = (definitions: Record<string, FlagDescriptor>): Set<string> => {
  const layer = readFlagOverrides(definitions);
  const envVars = new Set(Object.keys(layer));
  return new Set(
    Object.entries(definitions)
      .filter(([, descriptor]) => envVars.has(descriptor.envVar))
      .map(([key]) => key)
  );
};

const buildManifest = (
  definitions: Record<string, FlagDescriptor>,
  baseFlags: Record<string, unknown>,
  effectiveFlags: Record<string, unknown>,
  overridesEnabled: boolean
): FlagManifest => {
  const overridden = overridesEnabled ? overriddenKeys(definitions) : new Set<string>();

  return {
    version: FLAG_MANIFEST_VERSION,
    overridesEnabled,
    overridesKey: FEATURE_FLAG_OVERRIDES_KEY,
    flags: Object.entries(definitions).map(([key, descriptor]) => ({
      key,
      envVar: descriptor.envVar,
      type: descriptor.typeName,
      baseValue: baseFlags[key],
      effectiveValue: effectiveFlags[key],
      overridden: overridden.has(key),
    })),
  };
};

/**
 * Publish the manifest for the companion Chrome extension.
 *
 * When the gate is off a stub manifest is still written. Writing nothing would
 * leave the extension unable to distinguish "overrides are disabled in this
 * environment" from "this isn't a Studio page". This leaks nothing: every flag
 * key and `VITE_FF_*` name is already a literal string in the shipped bundle.
 */
export const writeFlagManifest = (
  definitions: Record<string, FlagDescriptor>,
  baseFlags: Record<string, unknown>,
  effectiveFlags: Record<string, unknown>,
  overridesEnabled: boolean
): void => {
  const storage = safeStorage();
  if (!storage) return;

  const manifest = overridesEnabled
    ? buildManifest(definitions, baseFlags, effectiveFlags, true)
    : { version: FLAG_MANIFEST_VERSION, overridesEnabled: false, overridesKey: '', flags: [] };

  try {
    storage.setItem(FEATURE_FLAG_MANIFEST_KEY, JSON.stringify(manifest));
  } catch {
    // Non-fatal: the extension simply won't find a manifest.
  }
};

/**
 * Persist the overrides blob. Returns false when storage rejected the write
 * (quota exceeded, storage blocked) so callers can report it rather than
 * throwing out of a devtools call and claiming a save that did not happen.
 */
const writeOverridesBlob = (blob: Record<string, string>): boolean => {
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

export type FlagsGlobal = {
  list: () => FlagManifestEntry[];
  manifest: () => FlagManifest;
  set: (key: string, value: string) => void;
  unset: (key: string) => void;
  reset: () => void;
};

/**
 * Attach `window.__flags` for humans working in devtools.
 *
 * This is deliberately *not* how the Chrome extension talks to the page — see
 * the module docstring. Writers refresh the manifest so a devtools edit and an
 * extension-driven edit converge on the same state.
 */
export const installFlagsGlobal = (
  definitions: Record<string, FlagDescriptor>,
  baseFlags: Record<string, unknown>,
  effectiveFlags: Record<string, unknown>,
  overridesEnabled: boolean
): void => {
  if (!safeStorage() || !overridesEnabled) return;

  const refresh = (saved: boolean) => {
    if (!saved) return;
    writeFlagManifest(definitions, baseFlags, effectiveFlags, overridesEnabled);
    logger.info('[featureFlags] Override saved. Reload the page to apply.');
  };

  const api: FlagsGlobal = {
    list: () => buildManifest(definitions, baseFlags, effectiveFlags, true).flags,
    manifest: () => buildManifest(definitions, baseFlags, effectiveFlags, true),
    set: (key, value) => {
      // Own-property check for the same reason as readFlagOverrides above.
      if (!Object.hasOwn(definitions, key)) {
        logger.error(`[featureFlags] Unknown flag "${key}".`);
        return;
      }
      const descriptor = definitions[key]!;
      if (!descriptor.schema.safeParse(value).success) {
        logger.error(`[featureFlags] "${value}" is not a valid ${descriptor.typeName}.`);
        return;
      }
      refresh(
        writeOverridesBlob({ ...(readOverridesBlob() as Record<string, string>), [key]: value })
      );
    },
    unset: (key) => {
      const blob = readOverridesBlob() as Record<string, string>;
      delete blob[key];
      refresh(writeOverridesBlob(blob));
    },
    reset: () => {
      refresh(writeOverridesBlob({}));
    },
  };

  (window as unknown as Record<string, unknown>).__flags = api;
};
