// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { logger } from '@nemo/common/src/utils/logger';
import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { PLUGINS_MANIFEST_ENDPOINT } from '@studio/plugins/consts';
import { isTrustedBundleUrl } from '@studio/plugins/security';
import type {
  LoadedPlugin,
  PluginManifest,
  PluginModule,
  PluginQueryData,
} from '@studio/plugins/types';

export function isValidPluginManifest(obj: unknown): obj is PluginManifest {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return typeof o.name === 'string' && (typeof o.bundleUrl === 'string' || o.bundleUrl === null);
}

export function isPluginModule(mod: unknown): mod is PluginModule {
  if (typeof mod !== 'object' || mod === null) return false;
  const m = mod as Record<string, unknown>;
  return typeof m.Root === 'function' && typeof m.navItems === 'function';
}

export async function loadPlugin(
  manifest: PluginManifest,
  baseUrl: string
): Promise<LoadedPlugin | null> {
  if (manifest.bundleUrl === null) {
    // Plugin registered without a web bundle — no UI to mount or nav to show.
    return null;
  }
  if (!isTrustedBundleUrl(manifest.bundleUrl)) {
    logger.warn(`[plugins] Rejected untrusted bundle URL: ${manifest.bundleUrl}`);
    return null;
  }
  // Prefix with the API base URL so the import resolves against the backend
  // host, not the Studio dev server (which may run on a different origin).
  const absoluteUrl = `${baseUrl}${manifest.bundleUrl}`;
  try {
    const module: unknown = await import(/* @vite-ignore */ absoluteUrl);
    if (!isPluginModule(module)) {
      logger.warn(`[plugins] Plugin "${manifest.name}" missing required exports (Root, navItems)`);
      return null;
    }
    return { name: manifest.name, Root: module.Root, navItems: module.navItems };
  } catch (err) {
    logger.warn(`[plugins] Failed to load plugin "${manifest.name}":`, err);
    return null;
  }
}

export async function fetchPlugins(): Promise<PluginQueryData> {
  // Falls back to same-origin when PLATFORM_BASE_URL is not configured
  const baseUrl = PLATFORM_BASE_URL ?? '';
  const res = await platformFetch(`${baseUrl}${PLUGINS_MANIFEST_ENDPOINT}`);
  if (!res.ok) throw new Error(`${PLUGINS_MANIFEST_ENDPOINT} returned ${res.status}`);
  const data: unknown = await res.json();
  if (!Array.isArray(data)) {
    // Throw (not empty-success) so a malformed manifest fails open like a
    // network error, rather than masquerading as "no plugins installed".
    logger.warn(`[plugins] ${PLUGINS_MANIFEST_ENDPOINT} did not return an array`);
    throw new Error(`${PLUGINS_MANIFEST_ENDPOINT} did not return an array`);
  }
  const invalid = (data as unknown[]).filter((item) => !isValidPluginManifest(item));
  if (invalid.length > 0) {
    logger.warn(
      `[plugins] ${PLUGINS_MANIFEST_ENDPOINT} returned ${invalid.length} invalid manifest(s) — skipping`
    );
  }
  const manifests = (data as unknown[]).filter(isValidPluginManifest);
  const loaded = await Promise.all(manifests.map((m) => loadPlugin(m, baseUrl)));
  return {
    installedNames: new Set(manifests.map((m) => m.name)),
    plugins: loaded.filter((p): p is LoadedPlugin => p !== null),
  };
}
