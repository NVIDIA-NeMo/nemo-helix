// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { LoadedPlugin, PluginLoadFailure } from '@studio/plugins/types';

/** Platform endpoint returning the installed plugin manifest. */
export const PLUGINS_MANIFEST_ENDPOINT = '/apis/plugins';

/** react-query key for the plugin manifest fetch. */
export const PLUGINS_MANIFEST_QUERY_KEY = ['plugins', 'manifest'] as const;

/**
 * Referentially-stable empty defaults so context consumers don't re-render on
 * every provider render before the manifest has loaded.
 */
export const NO_PLUGINS: LoadedPlugin[] = [];
export const NO_FAILURES: PluginLoadFailure[] = [];
export const NO_NAMES: ReadonlySet<string> = new Set();
