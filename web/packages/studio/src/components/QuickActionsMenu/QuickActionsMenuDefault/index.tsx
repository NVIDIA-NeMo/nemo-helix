// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  QuickActionItem,
  QuickActionsMenuRoot,
} from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { Pencil, Trash, FolderOpen } from '@nvidia/react-gui-icons/icons';
import { FC, useMemo } from 'react';
import { useNavigate } from 'react-router';

interface QuickActionsMenuDefaultProps {
  openTarget?: string;
  editAction?: () => void;
  deleteAction?: () => void;
}
/*
 * QuickActionsMenuDefault is a wrapper around QuickActionsMenu that provides presets for the
 * three most common actions used in these sorts of menus, along with default icons and danger
 * settings for each.
 */
export const QuickActionsMenuDefault: FC<QuickActionsMenuDefaultProps> = ({
  openTarget,
  editAction,
  deleteAction,
}) => {
  const navigate = useNavigate();
  const quickActions: QuickActionItem[] = useMemo(() => {
    const actions: QuickActionItem[] = [];
    if (openTarget) {
      actions.push({
        label: 'Open',
        onSelect: () => navigate(openTarget),
        icon: <FolderOpen />,
      });
    }
    if (editAction) {
      actions.push({
        label: 'Edit',
        onSelect: editAction,
        icon: <Pencil />,
      });
    }
    if (deleteAction) {
      actions.push({
        label: 'Delete',
        onSelect: deleteAction,
        icon: <Trash />,
        danger: true,
      });
    }
    return actions;
  }, [openTarget, editAction, deleteAction, navigate]);

  return <QuickActionsMenuRoot actions={quickActions} />;
};
