// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { pluginPath } from './paths';
import type { PluginNavGroup } from './types';

// The group label matches Studio's core group, so this row lands under
// Observability after Insights and Traces. The label is static because navItems
// runs before any data loads; the page picks the connection.
export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Observability',
    items: [
      {
        id: 'observability-trace-explorer',
        iconName: 'chart-flow',
        label: 'Trace Explorer',
        href: pluginPath(workspaceId),
      },
    ],
  },
];
