// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ContractIssue } from '@nemo/common/src/pluginContract/types';

export const CONTRACT_DOCS = 'plugins/example-plugin/web/AGENTS.md (in the nemo-helix repo)';

type Fix = Pick<ContractIssue, 'message' | 'steps'>;

const fix = (message: string, ...steps: string[]): Fix => ({
  message,
  steps: [...steps, `Contract and shared-dependency rules: ${CONTRACT_DOCS}`],
});

export const manifestMissing = (): Fix =>
  fix(
    'plugin.manifest.json was not found next to index.js.',
    "Add studioPlugin() from the Studio plugin build helper (studio-plugin.mjs) to the plugin's Vite config, then rebuild."
  );

export const specifierUnmapped = (specifier: string): Fix =>
  fix(
    `Studio does not provide "${specifier}" (it is not in the plugin contract). This can also mean the plugin was built with a modified or older build helper.`,
    `Import "${specifier}" as a normal dependency so the plugin bundles its own copy, then rebuild.`,
    `If it must be shared with Studio, request that "${specifier}" be added to the plugin contract.`
  );

// Rollup/Vite emit this deep import in development mode; Studio provides only the production runtime.
const DEV_RUNTIME = 'react/jsx-dev-runtime';

/** `specifier` is what the plugin imported, when it differs from the package (a deep import). */
export const singletonBundled = (pkg: string, specifier = pkg): Fix =>
  fix(
    `The singleton "${pkg}" is bundled into the plugin; a second copy breaks shared context.`,
    `Import "${pkg}" only by a contract specifier (e.g. react-router), never a deep path (e.g. react-router/dom) and never from node_modules directly. The build helper externalizes contract specifiers for you.`,
    ...(specifier === DEV_RUNTIME
      ? [
          `A development-mode build emits the dev JSX runtime ("${DEV_RUNTIME}"), which Studio does not provide. Build in production mode (e.g. vite build without --mode development).`,
        ]
      : []),
    'Rebuild the plugin.'
  );

/** A deep import of a provided package (e.g. `@nemo/common/components/Foo`). */
export const providedDeepImport = (specifier: string, pkg: string): Fix =>
  fix(
    `Studio does not provide "${specifier}" (it is not in the plugin contract); "${pkg}" is provided only through its bare specifier.`,
    `Import from the bare "${pkg}" specifier instead of "${specifier}", then rebuild.`,
    `If the name you need is not exported from "${pkg}", add it to web/packages/common/src/plugin.ts.`
  );

export const contractUnsupported = (built: string, current: string): Fix =>
  fix(
    `The plugin was built against contract "${built}", which Studio's contract ${current} does not support.`,
    `Update the plugin's copy of the build helper (studio-plugin.mjs) to the current one (contract ${current}), then rebuild.`
  );

export const versionIncompatible = (pkg: string, built: string, studio: string): Fix =>
  fix(
    `"${pkg}" was built against ${built}, but Studio ships ${studio} (different major version).`,
    `Set "${pkg}" to "${studio}" in the plugin's web/package.json, then rebuild.`
  );

/** Render an issue as build-error text: code, message, optional importer, numbered steps. */
export const formatIssue = (issue: ContractIssue, importer?: string): string =>
  [
    `${issue.code}: ${issue.message}`,
    ...(importer ? [`Imported from: ${importer}`] : []),
    ...issue.steps.map((step, i) => `${i + 1}. ${step}`),
  ].join('\n');
