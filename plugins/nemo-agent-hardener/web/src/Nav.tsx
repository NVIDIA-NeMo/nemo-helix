// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getAgentHardenerRunListRoute } from '@agent-hardener/paths';
import type { PluginNavGroup } from '@agent-hardener/types';

/**
 * Agent Hardener's entry in Studio's side nav. Studio renders these under the
 * named group, so installing the plugin is what puts the item there — Studio
 * carries no agent-hardener-specific nav code.
 */
export const navItems = (workspaceId: string): PluginNavGroup[] => [
  {
    group: 'Governance',
    items: [
      {
        id: 'agent-hardener',
        iconName: 'swords',
        label: 'Agent Hardener',
        href: getAgentHardenerRunListRoute(workspaceId),
      },
    ],
  },
];
