// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronDown, ChevronRight } from '@nemo/icons/icons';
import { Button, VerticalNavListItem, VerticalNavSubList } from '@nvidia/foundations-react-core';
import { StudioNavItem } from '@studio/components/Layouts/NavigationDrawer/components/StudioNavItem';
import { resolveActive } from '@studio/components/Layouts/NavigationDrawer/utils';
import { NavNotificationBadge } from '@studio/prototypes/notifications/components/NavNotificationBadge';
import { ProtoNavSubItem } from '@studio/prototypes/notifications/NavigationDrawer/ProtoNavSubItem';
import type { ProtoNavItem } from '@studio/prototypes/notifications/types';
import { rollupNotifications } from '@studio/prototypes/notifications/utils';
import type { FC } from 'react';
import { NavLink } from 'react-router';

interface ProtoExpandableNavItemProps {
  item: ProtoNavItem;
  isActive: (href: string) => boolean;
  accordionOpen: boolean | undefined;
  onAccordionChange: (itemId: string, open: boolean) => void;
}

export const ProtoExpandableNavItem: FC<ProtoExpandableNavItemProps> = ({
  item,
  isActive,
  accordionOpen,
  onAccordionChange,
}) => {
  const { href, subItems = [] } = item;
  const isOpen = accordionOpen ?? item.defaultOpen !== false;
  const subListId = `${item.id}-submenu`;
  const chevron = isOpen ? <ChevronDown /> : <ChevronRight />;
  const disclosure = { 'aria-expanded': isOpen, ...(isOpen && { 'aria-controls': subListId }) };
  const labelText = typeof item.slotLabel === 'string' ? item.slotLabel : undefined;
  const toggle = () => onAccordionChange(item.id, !isOpen);
  const active =
    resolveActive(item, isActive) ||
    (!isOpen && subItems.some((sub) => resolveActive(sub, isActive)));
  const notification = isOpen
    ? item.notification
    : rollupNotifications([item.notification, ...subItems.map((sub) => sub.notification)]);
  const badge = <NavNotificationBadge notification={notification} />;

  return (
    <VerticalNavListItem {...item.attributes?.VerticalNavListItem}>
      <div className="relative flex items-center">
        <StudioNavItem
          active={active}
          slotStart={item.slotIcon}
          slotEnd={
            href === undefined ? (
              <span className="ml-auto flex items-center gap-2">
                {badge}
                {chevron}
              </span>
            ) : (
              badge
            )
          }
          padding={href === undefined ? 'fullWidth' : 'trailingToggle'}
          {...item.attributes?.VerticalNavItem}
        >
          {href === undefined ? (
            <Button
              kind="tertiary"
              size="tiny"
              onClick={toggle}
              data-active-state="disabled"
              {...disclosure}
            >
              {item.slotLabel}
            </Button>
          ) : (
            <NavLink to={href}>{item.slotLabel}</NavLink>
          )}
        </StudioNavItem>
        {href !== undefined && (
          <Button
            kind="tertiary"
            size="tiny"
            data-active-state="disabled"
            onClick={toggle}
            className=" h-full px-3 py-1"
            aria-label={
              labelText ? `${isOpen ? 'Collapse' : 'Expand'} ${labelText}` : 'Toggle submenu'
            }
            {...disclosure}
          >
            {chevron}
          </Button>
        )}
      </div>
      {isOpen && (
        <VerticalNavSubList id={subListId} {...item.attributes?.VerticalNavSubList}>
          {subItems.map((sub) => (
            <ProtoNavSubItem key={sub.id} item={sub} isActive={isActive} />
          ))}
        </VerticalNavSubList>
      )}
    </VerticalNavListItem>
  );
};
