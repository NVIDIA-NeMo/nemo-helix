// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable unused-imports/no-unused-vars -- scaffold stub; remove with the implementation */
import type { Plugin } from 'rolldown';

export interface StudioPluginOptions {
  /** Directory whose `node_modules` holds the versions the plugin builds against. Defaults to cwd. */
  root?: string;
}

/**
 * Build plugin for Studio plugin bundles: externalizes the contract's shared
 * specifiers, fails the build when a singleton would be bundled, and emits
 * `plugin.manifest.json` next to the bundle.
 */
export const studioPlugin = (_options: StudioPluginOptions = {}): Plugin => ({
  name: 'nemo-studio-plugin',
  resolveId() {
    throw new Error('not implemented');
  },
});
