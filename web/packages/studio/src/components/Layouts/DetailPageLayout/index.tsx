// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Flex,
  PageHeader,
  Stack,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import type { ReactNode, Ref } from 'react';

export interface DetailPageTab<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface DetailPageLayoutProps<T extends string> {
  title: string;
  slotBadges?: ReactNode;
  slotDescription?: ReactNode;
  tabs: DetailPageTab<T>[];
  selectedTab: T;
  onTabChange: (tab: T) => void;
  slotActions?: ReactNode;
  tabsListRef?: Ref<HTMLDivElement>;
  children: ReactNode;
}

export const DetailPageLayout = <T extends string>({
  title,
  slotBadges,
  slotDescription,
  tabs,
  selectedTab,
  onTabChange,
  slotActions,
  tabsListRef,
  children,
}: DetailPageLayoutProps<T>) => (
  <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
    <PageHeader
      className="shrink-0 p-0"
      slotHeading={
        <Flex align="center" gap="3" className="min-w-0">
          <Text kind="title/md" className="min-w-0 truncate font-bold" title={title}>
            {title}
          </Text>
          {slotBadges && (
            <Flex align="center" gap="3" className="shrink-0">
              {slotBadges}
            </Flex>
          )}
        </Flex>
      }
      slotDescription={slotDescription}
    />

    <TabsRoot
      className="flex min-h-0 flex-1 flex-col"
      value={selectedTab}
      onValueChange={(value) => {
        const tab = tabs.find((candidate) => candidate.value === value);
        if (tab) {
          onTabChange(tab.value);
        }
      }}
    >
      <Flex className="shrink-0" align="center" gap="4">
        <TabsList className="min-w-0 flex-1" ref={tabsListRef}>
          {tabs.map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value} disabled={tab.disabled}>
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>
        {slotActions && (
          <Flex align="center" gap="2">
            {slotActions}
          </Flex>
        )}
      </Flex>
      {children}
    </TabsRoot>
  </Stack>
);
