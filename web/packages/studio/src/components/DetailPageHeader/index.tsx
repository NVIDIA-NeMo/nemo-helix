// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, TabsList, TabsTrigger, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

export interface DetailPageHeaderTab {
  value: string;
  label: ReactNode;
  disabled?: boolean;
}

export interface DetailPageHeaderProps {
  heading: ReactNode;
  /** Sits right after the heading, e.g. a status badge. */
  slotStatus?: ReactNode;
  /** Small read-only metadata badges after the status. */
  slotTags?: ReactNode;
  /** A line of text under the heading. */
  description?: ReactNode;
  tabs: DetailPageHeaderTab[];
  /** Actions for the selected tab, aligned right on the tab row. */
  slotActions?: ReactNode;
}

/**
 * Compact header for entity detail pages: identity on one row, tabs and the selected tab's
 * actions on the next. Renders the `TabsList`, so it must sit inside a `TabsRoot`; the page
 * owns the selected tab and renders the matching `TabsContent`.
 */
export const DetailPageHeader: FC<DetailPageHeaderProps> = ({
  heading,
  slotStatus,
  slotTags,
  description,
  tabs,
  slotActions,
}) => (
  <Stack gap="density-md" className="shrink-0">
    <Flex align="center" gap="density-md" wrap="wrap" className="min-w-0">
      <Text kind="title/sm" asChild className="min-w-0 truncate">
        <h1>{heading}</h1>
      </Text>
      {slotStatus}
      {slotTags && (
        <Flex align="center" gap="density-sm" wrap="wrap">
          {slotTags}
        </Flex>
      )}
    </Flex>
    {description && (
      <Text kind="body/regular/md" className="text-secondary">
        {description}
      </Text>
    )}
    <Flex align="end" gap="density-lg" className="min-w-0">
      <TabsList className="min-w-0 flex-1">
        {tabs.map((tab) => (
          <TabsTrigger key={tab.value} value={tab.value} disabled={tab.disabled}>
            {tab.label}
          </TabsTrigger>
        ))}
      </TabsList>
      {slotActions && (
        <Flex align="center" gap="density-sm" className="shrink-0">
          {slotActions}
        </Flex>
      )}
    </Flex>
  </Stack>
);
