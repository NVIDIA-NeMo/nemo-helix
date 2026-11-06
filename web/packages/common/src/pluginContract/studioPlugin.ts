// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { PLUGIN_CONTRACT } from '@nemo/common/src/pluginContract/contract';
import { issue } from '@nemo/common/src/pluginContract/diffContract';
import {
  formatIssue,
  singletonBundled,
  specifierUnmapped,
} from '@nemo/common/src/pluginContract/remediate';
import type { PluginSidecar } from '@nemo/common/src/pluginContract/types';
import fs from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'rolldown';

export interface StudioPluginOptions {
  /** Directory whose `node_modules` holds the versions the plugin builds against. Defaults to cwd. */
  root?: string;
}

const MANIFEST_FILE = 'plugin.manifest.json';
const { entries } = PLUGIN_CONTRACT;
const specifiers = new Set(entries.map((e) => e.specifier));
const packages = [...new Set(entries.map((e) => e.package))];
const singletons = new Set(entries.filter((e) => e.kind === 'singleton').map((e) => e.package));

const bundledSingletonIn = (moduleIds: Iterable<string>): string | undefined => {
  for (const id of moduleIds) {
    // Whole path segments only (`react-dom` must not match `react-dom-foo`); this also covers the
    // pnpm layout `node_modules/.pnpm/<x>/node_modules/<pkg>/`.
    const segments = id.replaceAll('\\', '/').split('/');
    for (
      let i = segments.indexOf('node_modules');
      i !== -1;
      i = segments.indexOf('node_modules', i + 1)
    ) {
      const name = segments[i + 1]?.startsWith('@')
        ? `${segments[i + 1]}/${segments[i + 2]}`
        : segments[i + 1];
      if (singletons.has(name)) {
        return name;
      }
    }
  }
  return undefined;
};

const readVersion = (root: string, pkg: string): string | undefined => {
  try {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(root, 'node_modules', pkg, 'package.json'), 'utf8')
    );
    return typeof manifest.version === 'string' ? manifest.version : undefined;
  } catch {
    return undefined;
  }
};

/**
 * Build plugin for Studio plugin bundles: externalizes the contract's shared
 * specifiers, fails the build when a singleton would be bundled, and emits
 * `plugin.manifest.json` next to the bundle.
 */
export const studioPlugin = ({ root = process.cwd() }: StudioPluginOptions = {}): Plugin & {
  enforce: 'pre';
} => ({
  name: 'nemo-studio-plugin',
  // Vite-only (rolldown ignores it): must run before Vite's core resolver, which would otherwise
  // resolve a contract specifier to a file in node_modules and bundle it.
  enforce: 'pre',

  // Singletons are checked twice: here, deep imports are caught with an actionable importer;
  // in generateBundle, source reached by paths that bypass resolution (relative node_modules).
  resolveId(source, importer) {
    if (source.startsWith('.') || path.isAbsolute(source)) {
      return null;
    }
    if (specifiers.has(source)) {
      return { id: source, external: true };
    }
    const owner = packages.find((pkg) => source.startsWith(`${pkg}/`));
    if (owner) {
      const found = singletons.has(owner)
        ? issue('PLUGIN_SINGLETON_BUNDLED', owner, singletonBundled(owner))
        : issue('PLUGIN_SPECIFIER_UNMAPPED', source, specifierUnmapped(source));
      this.error(formatIssue(found, importer));
    }
    return null;
  },

  generateBundle(_options, bundle) {
    const chunks = Object.values(bundle).filter((output) => output.type === 'chunk');

    const singleton = bundledSingletonIn(chunks.flatMap((chunk) => chunk.moduleIds));
    if (singleton) {
      this.error(
        formatIssue(issue('PLUGIN_SINGLETON_BUNDLED', singleton, singletonBundled(singleton)))
      );
    }

    // Chunk-to-chunk imports are bundle file names; everything else is a bare import.
    const bare = new Set(
      chunks
        .flatMap((chunk) => [...chunk.imports, ...chunk.dynamicImports])
        .filter((id) => !(id in bundle))
    );
    const builtAgainst: Record<string, string> = {};
    for (const pkg of packages) {
      const version = readVersion(root, pkg);
      if (version) {
        builtAgainst[pkg] = version;
      }
    }
    const manifest: PluginSidecar = {
      contractVersion: PLUGIN_CONTRACT.version,
      specifiers: [...bare].sort(),
      bundledPackages: [],
      builtAgainst,
    };
    this.emitFile({
      type: 'asset',
      fileName: MANIFEST_FILE,
      source: `${JSON.stringify(manifest, null, 2)}\n`,
    });
  },
});
