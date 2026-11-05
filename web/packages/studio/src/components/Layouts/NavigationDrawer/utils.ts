// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  NavGroup,
  NavInputItem,
  NavItem,
} from '@studio/components/Layouts/NavigationDrawer/types';

export const isGroup = (item: NavInputItem): item is { group?: string; items: NavItem[] } => {
  return Array.isArray((item as { items?: NavItem[] }).items);
};

/**
 * An item with no href is never active on its own — an expanded parent yields the highlight to
 * whichever child matched. `ExpandableNavItem` covers the collapsed case.
 */
export const resolveActive = (
  item: { active?: boolean; href?: string },
  isActive: (href: string) => boolean
): boolean => item.active ?? (item.href !== undefined ? isActive(item.href) : false);

/** Normalize input into groups. Ungrouped items become their own single-item group. */
export const toGroups = (items: NavInputItem[]): NavGroup[] => {
  const groups: NavGroup[] = [];
  for (const entry of items) {
    if (isGroup(entry)) {
      groups.push({ groupLabel: entry.group, items: entry.items });
    } else {
      groups.push({ items: [entry] });
    }
  }
  return groups;
};

/**
 * Flatten parents and children for the collapsed rail, which has no room for accordions. The rail
 * is nothing but links, so anything without an href drops out.
 */
export const flattenForRail = (items: NavItem[]): NavItem[] =>
  items.flatMap((item) => [
    ...(item.href !== undefined ? [{ ...item, subItems: undefined }] : []),
    ...(item.subItems ?? []).filter((sub) => sub.href !== undefined),
  ]);
