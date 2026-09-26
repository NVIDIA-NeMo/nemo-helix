// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pluginPath } from './paths';
import type { PluginNavGroup } from './types';

/**
 * Returns the nav groups for the kern plugin.
 *
 * One group "Kern", one item "Dashboard". The href is an absolute path so
 * Studio's side nav renders it as a direct link without relative-path
 * appending.
 */
export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Kern',
    items: [
      {
        id: 'kern-dashboard',
        iconName: 'shield',
        label: 'Dashboard',
        href: pluginPath(workspaceId, 'dashboard'),
      },
    ],
  },
];
