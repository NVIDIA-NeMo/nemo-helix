// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pluginPath } from './paths';
import type { PluginNavGroup } from './types';

export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Christmas',
    items: [
      {
        id: 'is-it-christmas',
        iconName: 'calendar',
        label: 'Is it Christmas?',
        href: pluginPath(workspaceId),
      },
    ],
  },
];
