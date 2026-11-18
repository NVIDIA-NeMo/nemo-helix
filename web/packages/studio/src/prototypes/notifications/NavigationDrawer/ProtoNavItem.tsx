// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ProtoExpandableNavItem } from '@studio/prototypes/notifications/NavigationDrawer/ProtoExpandableNavItem';
import { ProtoNavLinkItem } from '@studio/prototypes/notifications/NavigationDrawer/ProtoNavLinkItem';
import type { ProtoNavItem as ProtoNavItemData } from '@studio/prototypes/notifications/types';
import type { FC } from 'react';

interface ProtoNavItemProps {
  item: ProtoNavItemData;
  isActive: (href: string) => boolean;
  accordionOpen: boolean | undefined;
  onAccordionChange: (itemId: string, open: boolean) => void;
}

export const ProtoNavItem: FC<ProtoNavItemProps> = ({
  item,
  isActive,
  accordionOpen,
  onAccordionChange,
}) =>
  item.subItems?.length ? (
    <ProtoExpandableNavItem
      item={item}
      isActive={isActive}
      accordionOpen={accordionOpen}
      onAccordionChange={onAccordionChange}
    />
  ) : (
    <ProtoNavLinkItem item={item} isActive={isActive} />
  );
