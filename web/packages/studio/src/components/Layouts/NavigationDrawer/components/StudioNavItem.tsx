// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { VerticalNavItem } from '@nvidia/foundations-react-core';
import {
  ACTIVE_NAV_ITEM_CLASS,
  COMPACT_NAV_ROW_CLASS,
  NAV_ROW_PADDING,
} from '@studio/components/Layouts/NavigationDrawer/styles';
import cn from 'classnames';
import type { ComponentPropsWithoutRef, FC, ReactNode } from 'react';

type KuiVerticalNavItemProps = Omit<
  ComponentPropsWithoutRef<typeof VerticalNavItem>,
  'children' | 'asChild'
>;

interface StudioNavItemProps extends KuiVerticalNavItemProps {
  padding?: keyof typeof NAV_ROW_PADDING;
  /** The element this row renders as — a NavLink, or a <button> for disclosure rows. */
  children: ReactNode;
}

/**
 * Every row in the drawer, expanded or collapsed. Renders `asChild` so the row *is* its link or
 * button, and merges a caller's `className` after the defaults instead of spreading over them.
 */
export const StudioNavItem: FC<StudioNavItemProps> = ({
  active,
  padding = 'default',
  className,
  children,
  ...rest
}) => {
  const classes =
    cn(
      COMPACT_NAV_ROW_CLASS,
      NAV_ROW_PADDING[padding],
      active && ACTIVE_NAV_ITEM_CLASS,
      className
    ) || undefined;

  return (
    <VerticalNavItem active={active} {...rest} className={classes} asChild>
      {children}
    </VerticalNavItem>
  );
};
