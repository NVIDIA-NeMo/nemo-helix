// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pluginPath } from './paths';
import type { PluginNavGroup } from './types';

/**
 * Returns the nav groups for this plugin.
 *
 * Each item's `href` is an absolute path that Studio's side nav renders as a
 * link.  Sub-pages are just deeper paths within the plugin's route subtree
 * (`/workspaces/:id/plugin/example/*`).
 */
export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Example Plugin',
    items: [
      {
        id: 'example-overview',
        iconName: 'beaker',
        label: 'Overview',
        href: pluginPath(workspaceId, 'overview'),
      },
      {
        id: 'example-auth',
        iconName: 'key',
        label: 'Auth',
        href: pluginPath(workspaceId, 'auth'),
      },
      {
        id: 'example-workspace',
        iconName: 'business',
        label: 'Workspace',
        href: pluginPath(workspaceId, 'workspace'),
      },
      {
        id: 'example-shared-ui',
        iconName: 'layout-grid',
        label: 'Shared UI',
        href: pluginPath(workspaceId, 'shared-ui'),
      },
    ],
  },
];
