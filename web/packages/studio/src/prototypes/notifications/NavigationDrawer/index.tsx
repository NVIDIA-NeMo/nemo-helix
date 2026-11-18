// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  VerticalNavGroup,
  VerticalNavGroupList,
  VerticalNavHeading,
  VerticalNavList,
  VerticalNavRoot,
} from '@nvidia/foundations-react-core';
import { ProtoCollapsedNavItem } from '@studio/prototypes/notifications/NavigationDrawer/ProtoCollapsedNavItem';
import { ProtoNavItem } from '@studio/prototypes/notifications/NavigationDrawer/ProtoNavItem';
import type { ProtoNavGroup, ProtoNavInputItem } from '@studio/prototypes/notifications/types';
import { flattenProtoForRail, toProtoGroups } from '@studio/prototypes/notifications/utils';
import { Fragment, useCallback, useMemo, useState, type FC } from 'react';
import { useLocation } from 'react-router';

interface ProtoNavigationDrawerProps {
  items: ProtoNavInputItem[];
  collapsed?: boolean;
}

export const ProtoNavigationDrawer: FC<ProtoNavigationDrawerProps> = ({
  items,
  collapsed = false,
}) => {
  const { pathname } = useLocation();
  const [accordionState, setAccordionState] = useState<Record<string, boolean>>({});

  const groups = useMemo(() => toProtoGroups(items), [items]);

  const matchedHref = useMemo(() => {
    const hrefs = groups
      .flatMap((g) => g.items)
      .flatMap((i) => [i.href, ...(i.subItems ?? []).map((s) => s.href)])
      .filter((h): h is string => typeof h === 'string');
    const matching = hrefs.filter((h) => pathname === h || pathname.startsWith(`${h}/`));
    return matching.reduce<string | null>(
      (best, h) => (best === null || h.length > best.length ? h : best),
      null
    );
  }, [groups, pathname]);

  const isActive = useCallback((href: string) => href === matchedHref, [matchedHref]);

  const handleAccordionChange = (itemId: string, open: boolean) => {
    setAccordionState((prev) => ({ ...prev, [itemId]: open }));
  };

  const renderGroups = (groupList: ProtoNavGroup[]) =>
    groupList
      .map((group) => ({
        groupLabel: group.groupLabel,
        items: collapsed ? flattenProtoForRail(group.items) : group.items,
      }))
      .filter((group) => group.items.length > 0)
      .map((group, groupIndex) => {
        const navItems = group.items.map((item) =>
          collapsed ? (
            <ProtoCollapsedNavItem key={item.id} item={item} isActive={isActive} />
          ) : (
            <ProtoNavItem
              key={item.id}
              item={item}
              isActive={isActive}
              accordionOpen={accordionState[item.id]}
              onAccordionChange={handleAccordionChange}
            />
          )
        );

        if (group.groupLabel === undefined || (collapsed && groupIndex === 0)) {
          return <Fragment key={groupIndex}>{navItems}</Fragment>;
        }

        return (
          <VerticalNavGroup key={groupIndex}>
            <VerticalNavHeading className="h-8 min-h-8">{group.groupLabel}</VerticalNavHeading>
            <VerticalNavGroupList>{navItems}</VerticalNavGroupList>
          </VerticalNavGroup>
        );
      });

  return (
    <VerticalNavRoot
      collapsed={collapsed}
      className="overflow-y-auto overflow-x-hidden transition-[width] duration-200"
    >
      <VerticalNavList className="px-2 py-3">{renderGroups(groups)}</VerticalNavList>
    </VerticalNavRoot>
  );
};
