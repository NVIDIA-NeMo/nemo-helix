// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getRunListRoute } from './paths';
import type { PluginNavGroup } from './types';

export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'User Simulation',
    items: [
      {
        id: 'user-sim-runs',
        iconName: 'users',
        label: 'Simulations',
        href: getRunListRoute(workspaceId),
      },
    ],
  },
];
