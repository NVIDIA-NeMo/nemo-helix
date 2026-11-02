// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tooltip, VerticalNavListItem } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import type { NavItem as NavItemData } from '@studio/components/Layouts/NavigationDrawer/types';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface CollapsedNavItemProps {
  item: NavItemData;
  isActive: (href: string) => boolean;
}

export const CollapsedNavItem: FC<CollapsedNavItemProps> = ({ item, isActive }) => {
  const { href } = item;
  if (href === undefined) return null;

  // The rail is icon-only, so an item without one falls back to its initial rather than vanishing.
  const icon = item.slotIcon ?? (
    <span aria-hidden className="w-4 text-center text-xs">
      {typeof item.slotLabel === 'string' ? item.slotLabel.charAt(0).toUpperCase() : '•'}
    </span>
  );

  return (
    <VerticalNavListItem {...item.attributes?.VerticalNavListItem}>
      <Tooltip slotContent={item.slotLabel} side="right">
        <StudioNavItem
          active={resolveActive(item, isActive)}
          slotStart={icon}
          {...item.attributes?.VerticalNavItem}
        >
          <NavLink to={href}>{item.slotLabel}</NavLink>
        </StudioNavItem>
      </Tooltip>
    </VerticalNavListItem>
  );
};
