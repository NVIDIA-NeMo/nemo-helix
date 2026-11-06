// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ContractIssue } from '@nemo/common/src/pluginContract/types';

const CONTRACT_DOCS = 'plugins/example-plugin/web/AGENTS.md';

type Fix = Pick<ContractIssue, 'message' | 'steps'>;

const fix = (message: string, ...steps: string[]): Fix => ({
  message,
  steps: [...steps, `Contract and shared-dependency rules: ${CONTRACT_DOCS}`],
});

export const manifestMissing = (): Fix =>
  fix(
    'plugin.manifest.json was not found next to index.js.',
    'Rebuild the plugin with the Studio plugin build helper so it emits plugin.manifest.json next to index.js.'
  );

export const specifierUnmapped = (specifier: string): Fix =>
  fix(
    `The plugin imports "${specifier}" at runtime, but it is not in the plugin contract.`,
    `To bundle it, remove "${specifier}" from build.rolldownOptions.external in the plugin build config, then run pnpm build.`,
    `If it must be shared with Studio, request that "${specifier}" be added to the plugin contract (see ${CONTRACT_DOCS}).`
  );

export const singletonBundled = (pkg: string): Fix =>
  fix(
    `The singleton "${pkg}" is bundled into the plugin; a second copy breaks shared context.`,
    `Ensure "${pkg}" is listed in build.rolldownOptions.external in the plugin build config.`,
    `Import "${pkg}" only by its contract specifier, not by a deep path, then run pnpm build.`
  );

export const contractUnsupported = (built: string, current: string): Fix =>
  fix(
    `The plugin was built against contract "${built}", which Studio's contract ${current} does not support.`,
    `Rebuild the plugin against the current contract (${current}) with pnpm build.`
  );

export const versionIncompatible = (pkg: string, built: string, studio: string): Fix =>
  fix(
    `"${pkg}" was built against ${built}, but Studio ships ${studio} (different major version).`,
    `Set "${pkg}" to "${studio}" in the plugin's web/package.json, then rebuild with pnpm build.`
  );
