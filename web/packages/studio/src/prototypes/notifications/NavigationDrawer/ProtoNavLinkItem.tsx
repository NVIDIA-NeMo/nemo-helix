// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VerticalNavListItem } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import { NavNotificationBadge } from '@studio/prototypes/notifications/components/NavNotificationBadge';
import type { ProtoNavItem } from '@studio/prototypes/notifications/types';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface ProtoNavLinkItemProps {
  item: ProtoNavItem;
  isActive: (href: string) => boolean;
}

export const ProtoNavLinkItem: FC<ProtoNavLinkItemProps> = ({ item, isActive }) => {
  const { href } = item;
  if (href === undefined) {
    return null;
  }

  return (
    <VerticalNavListItem {...item.attributes?.VerticalNavListItem}>
      <StudioNavItem
        active={resolveActive(item, isActive)}
        slotStart={item.slotIcon}
        slotEnd={<NavNotificationBadge notification={item.notification} />}
        {...item.attributes?.VerticalNavItem}
      >
        <NavLink to={href}>{item.slotLabel}</NavLink>
      </StudioNavItem>
    </VerticalNavListItem>
  );
};
