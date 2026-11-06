// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLUGIN_CONTRACT, diffContract, type PluginSidecar } from '@nemo/common/src/pluginContract';
import { logger } from '@nemo/common/src/utils/logger';
import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { PLUGINS_MANIFEST_ENDPOINT } from '@studio/plugins/consts';
import { reportLoadFailures } from '@studio/plugins/reportLoadFailures';
import { isTrustedBundleUrl } from '@studio/plugins/security';
import { STUDIO_SHARED_VERSIONS } from '@studio/plugins/studioVersions';
import type {
  PluginLoadResult,
  PluginManifest,
  PluginModule,
  PluginQueryData,
} from '@studio/plugins/types';

const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err));

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

const fail = (name: string, message: string): PluginLoadResult => {
  logger.warn(`[plugins] Plugin "${name}" failed to load: ${message}`);
  return {
    failure: {
      name,
      issues: [
        {
          code: 'PLUGIN_LOAD_FAILED',
          subject: '',
          message,
          steps: [
            'Check the browser console for the underlying error.',
            'Rebuild the plugin and refresh Studio.',
            'Contract and shared-dependency rules: plugins/example-plugin/web/AGENTS.md',
          ],
        },
      ],
    },
  };
};

const isSidecar = (obj: unknown): obj is PluginSidecar => {
  if (typeof obj !== 'object' || obj === null) return false;
  const o = obj as Record<string, unknown>;
  const isStrings = (v: unknown) => Array.isArray(v) && v.every((x) => typeof x === 'string');
  return (
    typeof o.contractVersion === 'string' &&
    isStrings(o.specifiers) &&
    isStrings(o.bundledPackages) &&
    typeof o.builtAgainst === 'object' &&
    o.builtAgainst !== null
  );
};

export async function loadPlugin(
  manifest: PluginManifest,
  baseUrl: string,
  importModule: (url: string) => Promise<unknown> = (url) => import(/* @vite-ignore */ url)
): Promise<PluginLoadResult> {
  const { name } = manifest;
  if (manifest.bundleUrl === null) {
    // Plugin registered without a web bundle — no UI to mount or nav to show.
    return null;
  }
  if (!isTrustedBundleUrl(manifest.bundleUrl)) {
    return fail(name, `Rejected untrusted bundle URL: ${manifest.bundleUrl}`);
  }
  // Prefix with the API base URL so the import resolves against the backend
  // host, not the Studio dev server (which may run on a different origin).
  const absoluteUrl = `${baseUrl}${manifest.bundleUrl}`;
  // The sidecar sits next to the validated bundle.
  const sidecarUrl = absoluteUrl.replace(/[^/]*$/, 'plugin.manifest.json');
  let sidecar: PluginSidecar | null;
  try {
    const res = await platformFetch(sidecarUrl);
    if (res.status === 404) {
      sidecar = null;
    } else if (!res.ok) {
      return fail(name, `${sidecarUrl} returned ${res.status}`);
    } else {
      const body: unknown = await res.json();
      if (!isSidecar(body)) return fail(name, `${sidecarUrl} is not a valid plugin manifest`);
      sidecar = body;
    }
  } catch (err) {
    return fail(name, `Could not read ${sidecarUrl}: ${errorMessage(err)}`);
  }
  const issues = diffContract(sidecar, PLUGIN_CONTRACT, STUDIO_SHARED_VERSIONS);
  if (issues.length > 0) {
    logger.warn(`[plugins] Plugin "${name}" violates the plugin contract:`, issues);
    return { failure: { name, issues } };
  }
  try {
    const module: unknown = await importModule(absoluteUrl);
    if (!isPluginModule(module))
      return fail(name, 'Bundle is missing required exports (Root, navItems)');
    return { plugin: { name, Root: module.Root, navItems: module.navItems } };
  } catch (err) {
    return fail(name, `Bundle failed to load: ${errorMessage(err)}`);
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
  const failed = loaded.flatMap((r) => (r && 'failure' in r ? [r.failure] : []));
  if (failed.length > 0) void reportLoadFailures(baseUrl, failed);
  return {
    installedNames: new Set(manifests.map((m) => m.name)),
    plugins: loaded.flatMap((r) => (r && 'plugin' in r ? [r.plugin] : [])),
    failed,
  };
}
