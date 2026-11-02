// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  VerticalNavGroup,
  VerticalNavGroupList,
  VerticalNavHeading,
  VerticalNavList,
  VerticalNavRoot,
} from '@nvidia/foundations-react-core';
import { CollapsedNavItem } from '@studio/components/Layouts/NavigationDrawer/components/CollapsedNavItem';
import { NavItem } from '@studio/components/Layouts/NavigationDrawer/components/NavItem';
import { Props } from '@studio/components/Layouts/NavigationDrawer/types';
import { flattenForRail, toGroups } from '@studio/components/Layouts/NavigationDrawer/utils';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { NAV_ACCORDION_STATE_KEY } from '@studio/util/localStorage';
import { Fragment, useCallback, useMemo, type FC } from 'react';
import { useLocation } from 'react-router';

export const NavigationDrawer: FC<Props> = ({ items, collapsed = false }) => {
  const { pathname } = useLocation();
  // Each header's open/closed choice belongs to the user: persisted per id in localStorage, it
  // outlives navigation and the browser session, and only a click ever changes it. Headers the
  // user has never touched fall through to `defaultOpen`, which is open unless an item opts out.
  const [accordionState = {}, setAccordionState] = useLocalStorage<Record<string, boolean>>(
    NAV_ACCORDION_STATE_KEY,
    {}
  );

  const groups = useMemo(() => toGroups(items), [items]);

  // Active item = the nav href that is the longest prefix of the current path.
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
    setAccordionState({ ...accordionState, [itemId]: open });
  };

  const renderGroups = (groupList: ReturnType<typeof toGroups>) =>
    groupList
      .map((group) => ({
        groupLabel: group.groupLabel,
        items: collapsed ? flattenForRail(group.items) : group.items,
      }))
      // Resolve items before the heading, so a group the rail empties out takes its heading too.
      .filter((group) => group.items.length > 0)
      .map((group, groupIndex) => {
        const navItems = group.items.map((item) =>
          collapsed ? (
            <CollapsedNavItem key={item.id} item={item} isActive={isActive} />
          ) : (
            <NavItem
              key={item.id}
              item={item}
              isActive={isActive}
              accordionOpen={accordionState[item.id]}
              onAccordionChange={handleAccordionChange}
            />
          )
        );

        // KUI draws a collapsed heading as a rule, which would open the rail with a stray line.
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
      <VerticalNavList className="p-3">{renderGroups(groups)}</VerticalNavList>
    </VerticalNavRoot>
  );
};
