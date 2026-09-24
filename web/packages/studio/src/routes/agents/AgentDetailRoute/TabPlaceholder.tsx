// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { ComponentType, FC } from 'react';

interface TabPlaceholderProps {
  icon: ComponentType<{ className?: string }>;
  title: string;
  description: string;
}

/** Empty-state card for tabs whose backing data/experience isn't built yet. */
export const TabPlaceholder: FC<TabPlaceholderProps> = ({ icon: Icon, title, description }) => (
  <Flex
    align="center"
    justify="center"
    className="min-h-full w-full rounded-xl border border-dashed border-base bg-surface-raised px-6 py-16"
  >
    <Stack gap="2" align="center" className="max-w-md text-center">
      <Flex
        align="center"
        justify="center"
        className="size-11 rounded-full bg-surface-subtle text-secondary"
      >
        <Icon className="size-12" aria-hidden />
      </Flex>
      <Text kind="body/bold/md">{title}</Text>
      <Text className="text-secondary" kind="body/regular/sm">
        {description}
      </Text>
    </Stack>
  </Flex>
);
