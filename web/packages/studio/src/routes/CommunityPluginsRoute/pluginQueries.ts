// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getInstalledApisPluginsInstalledGetQueryKey } from '@nemo/sdk/generated/platform/external-plugins';
import { PLUGINS_MANIFEST_QUERY_KEY } from '@studio/plugins/consts';
import type { QueryClient } from '@tanstack/react-query';

/**
 * Registering, enabling or removing a plugin changes two caches: this page's registry
 * list, and the manifest Studio loads plugin bundles and the side nav from. The second
 * is what makes a change show up in the nav without a page reload.
 */
export const invalidatePluginCaches = (queryClient: QueryClient): Promise<unknown> =>
  Promise.all([
    queryClient.invalidateQueries({ queryKey: getInstalledApisPluginsInstalledGetQueryKey() }),
    queryClient.invalidateQueries({ queryKey: PLUGINS_MANIFEST_QUERY_KEY }),
  ]);
