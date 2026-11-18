// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VerticalNavSubListItem } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import { NavNotificationBadge } from '@studio/prototypes/notifications/components/NavNotificationBadge';
import type { ProtoNavSubItem as ProtoNavSubItemData } from '@studio/prototypes/notifications/types';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface ProtoNavSubItemProps {
  item: ProtoNavSubItemData;
  isActive: (href: string) => boolean;
}

export const ProtoNavSubItem: FC<ProtoNavSubItemProps> = ({ item, isActive }) => {
  const { href } = item;

  return (
    <VerticalNavSubListItem {...item.attributes?.VerticalNavSubListItem}>
      <StudioNavItem
        kind="secondary"
        active={resolveActive(item, isActive)}
        disabled={!href}
        slotStart={item.slotIcon}
        slotEnd={<NavNotificationBadge notification={item.notification} />}
        {...item.attributes?.VerticalNavItem}
      >
        {href ? <NavLink to={href}>{item.slotLabel}</NavLink> : <span>{item.slotLabel}</span>}
      </StudioNavItem>
    </VerticalNavSubListItem>
  );
};
