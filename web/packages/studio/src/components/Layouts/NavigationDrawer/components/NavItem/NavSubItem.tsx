// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VerticalNavSubListItem } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import type { NavSubItem as NavSubItemData } from '@studio/components/Layouts/NavigationDrawer/types';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface NavSubItemProps {
  item: NavSubItemData;
  isActive: (href: string) => boolean;
}

export const NavSubItem: FC<NavSubItemProps> = ({ item, isActive }) => {
  const { href } = item;

  return (
    <VerticalNavSubListItem {...item.attributes?.VerticalNavSubListItem}>
      <StudioNavItem
        kind="secondary"
        active={resolveActive(item, isActive)}
        disabled={!href}
        slotStart={item.slotIcon}
        {...item.attributes?.VerticalNavItem}
      >
        {/* Under `asChild` the child is ours, so KUI cannot swap in its own <span> for a disabled
            row — a sub-item with nowhere to go supplies the span itself. */}
        {href ? <NavLink to={href}>{item.slotLabel}</NavLink> : <span>{item.slotLabel}</span>}
      </StudioNavItem>
    </VerticalNavSubListItem>
  );
};
