// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tooltip, VerticalNavListItem } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import { NavNotificationBadge } from '@studio/prototypes/notifications/components/NavNotificationBadge';
import { NavNotificationDot } from '@studio/prototypes/notifications/components/NavNotificationDot';
import type { ProtoNavItem } from '@studio/prototypes/notifications/types';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface ProtoCollapsedNavItemProps {
  item: ProtoNavItem;
  isActive: (href: string) => boolean;
}

export const ProtoCollapsedNavItem: FC<ProtoCollapsedNavItemProps> = ({ item, isActive }) => {
  const { href } = item;
  if (href === undefined) {
    return null;
  }

  const icon = item.slotIcon ?? (
    <span aria-hidden className="w-4 text-center text-xs">
      {typeof item.slotLabel === 'string' ? item.slotLabel.charAt(0).toUpperCase() : '•'}
    </span>
  );

  return (
    <VerticalNavListItem {...item.attributes?.VerticalNavListItem}>
      <Tooltip
        slotContent={
          <span className="flex items-center gap-2">
            {item.slotLabel}
            <NavNotificationBadge notification={item.notification} />
          </span>
        }
        side="right"
      >
        <StudioNavItem
          active={resolveActive(item, isActive)}
          slotStart={
            <NavNotificationDot notification={item.notification}>{icon}</NavNotificationDot>
          }
          {...item.attributes?.VerticalNavItem}
        >
          <NavLink to={href}>{item.slotLabel}</NavLink>
        </StudioNavItem>
      </Tooltip>
    </VerticalNavListItem>
  );
};
