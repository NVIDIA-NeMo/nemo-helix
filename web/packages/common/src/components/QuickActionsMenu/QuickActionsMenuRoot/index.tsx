// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ActionsMenu, type ActionMenuItem } from '@nemo/common/src/components/ActionsMenu';
import { MoreVert } from '@nvidia/react-gui-icons/icons';
import type { FC } from 'react';

export type QuickActionItem = ActionMenuItem;

interface QuickActionsMenuProps {
  actions: QuickActionItem[];
}

/*
 * QuickActionsMenu is the icon-only preset of ActionsMenu, used for row- and card-level menus.
 */
export const QuickActionsMenuRoot: FC<QuickActionsMenuProps> = ({ actions }) => (
  <ActionsMenu
    actions={actions}
    icon={<MoreVert />}
    ariaLabel="Open quick actions menu"
    data-testid="quick-actions-menu"
  />
);
