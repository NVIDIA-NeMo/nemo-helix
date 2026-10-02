// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Divider,
  Text,
  VerticalNavList,
  VerticalNavListItem,
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
  // user has never touched fall through to `defaultOpen`, which opens the current section.
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

  // The rail has no room for a heading, so a group announces itself as a rule between icon
  // sections instead — named, since collapsing drops the only other clue about the grouping.
  const renderGroupLabel = (groupLabel: string, groupIndex: number) =>
    collapsed ? (
      groupIndex > 0 && (
        <VerticalNavListItem className="px-2 py-2">
          <Divider aria-label={groupLabel} />
        </VerticalNavListItem>
      )
    ) : (
      <VerticalNavListItem className={groupIndex > 0 ? 'py-1 px-3 mt-4' : 'px-3'}>
        <Text kind="body/semibold/sm" className="text-subtle">
          {groupLabel}
        </Text>
      </VerticalNavListItem>
    );

  const renderGroups = (groupList: ReturnType<typeof toGroups>) =>
    groupList
      .map((group) => ({
        groupLabel: group.groupLabel,
        items: collapsed ? flattenForRail(group.items) : group.items,
      }))
      // Resolve items before the heading, so a group the rail empties out takes its heading too.
      .filter((group) => group.items.length > 0)
      .map((group, groupIndex) => (
        <Fragment key={groupIndex}>
          {group.groupLabel !== undefined && renderGroupLabel(group.groupLabel, groupIndex)}
          {collapsed
            ? group.items.map((item) => (
                <CollapsedNavItem key={item.id} item={item} isActive={isActive} />
              ))
            : group.items.map((item) => (
                <NavItem
                  key={item.id}
                  item={item}
                  isActive={isActive}
                  accordionOpen={accordionState[item.id]}
                  onAccordionChange={handleAccordionChange}
                />
              ))}
        </Fragment>
      ));

  return (
    <VerticalNavRoot
      className={`overflow-y-auto overflow-x-hidden transition-[width] duration-200 ${collapsed ? 'w-12' : 'w-60'}`}
    >
      <VerticalNavList className={`${collapsed ? 'py-3' : 'p-3'}`}>
        {renderGroups(groups)}
      </VerticalNavList>
    </VerticalNavRoot>
  );
};
