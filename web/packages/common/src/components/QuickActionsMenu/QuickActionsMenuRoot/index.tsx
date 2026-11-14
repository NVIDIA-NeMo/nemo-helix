// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ActionsMenu,
  type ActionMenuItem,
  type DeleteAction,
} from '@nemo/common/src/components/ActionsMenu';
import { MoreVert } from '@nemo/icons/icons';
import type { FC } from 'react';

export type QuickActionItem = ActionMenuItem;

interface QuickActionsMenuProps {
  actions: QuickActionItem[];
  deleteAction?: DeleteAction;
  ariaLabel?: string;
  disabled?: boolean;
}

/*
 * QuickActionsMenu is the icon-only preset of ActionsMenu, used for row- and card-level menus.
 */
export const QuickActionsMenuRoot: FC<QuickActionsMenuProps> = ({
  actions,
  deleteAction,
  ariaLabel = 'Open quick actions menu',
  disabled,
}) => (
  <ActionsMenu
    actions={actions}
    deleteAction={deleteAction}
    icon={<MoreVert />}
    ariaLabel={ariaLabel}
    disabled={disabled}
    data-testid="quick-actions-menu"
  />
);
