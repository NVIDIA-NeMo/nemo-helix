// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The shared-dependency contract between Studio and its runtime-loaded plugins.
//
// Two properties are checked and they differ: compatibility (is the API version
// Studio provides one the plugin was built for?) and identity (does the plugin
// resolve a singleton to Studio's instance instead of bundling its own?).

/**
 * singleton: identity required; a second copy breaks shared context (react-router).
 * provided: Studio serves the API, but the plugin's copy has its own module state.
 */
export type ContractEntryKind = 'singleton' | 'provided';

export interface ContractEntry {
  /** The bare import specifier a plugin leaves unbundled, e.g. `react/jsx-runtime`. */
  specifier: string;
  /** The npm package that owns the specifier, e.g. `react`. */
  package: string;
  kind: ContractEntryKind;
}

export interface PluginContract {
  /**
   * Contract semver. Major bump: an entry was removed or its kind changed (breaks existing
   * plugins). Minor bump: an entry was added (plugins built against an older minor still load;
   * a plugin built against a newer minor than Studio's is unsupported).
   */
  version: string;
  entries: readonly ContractEntry[];
}

/** Emitted by the plugin build next to `index.js` as `plugin.manifest.json`. */
export interface PluginSidecar {
  /** The contract version the plugin was built against. */
  contractVersion: string;
  /** Bare imports the built bundle leaves for the browser to resolve. */
  specifiers: string[];
  /** Packages whose source was found bundled into the output. */
  bundledPackages: string[];
  /** Package name to the version the plugin built (typed) against. */
  builtAgainst: Record<string, string>;
}

export type IssueCode =
  | 'PLUGIN_MANIFEST_MISSING'
  | 'PLUGIN_SPECIFIER_UNMAPPED'
  | 'PLUGIN_SINGLETON_BUNDLED'
  | 'PLUGIN_CONTRACT_UNSUPPORTED'
  | 'PLUGIN_VERSION_INCOMPATIBLE';

export interface ContractIssue {
  code: IssueCode;
  /** The package or specifier the issue is about; empty for a missing sidecar. */
  subject: string;
  message: string;
  /** Ordered resolution steps naming the exact change to make. */
  steps: string[];
}
