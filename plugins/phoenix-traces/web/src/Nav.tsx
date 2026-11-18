// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pluginPath } from './paths';
import type { PluginNavGroup } from './types';

export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Phoenix',
    items: [
      {
        id: 'phoenix-traces',
        iconName: 'workflow',
        label: 'Traces',
        href: pluginPath(workspaceId, 'traces'),
      },
    ],
  },
];
