// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

import type { MenuItem, ReportSection } from '..';

interface ReportMenuProps {
  onSectionChange?: (section: ReportSection) => void;
  items: MenuItem[];
}

export const ReportMenu: FC<ReportMenuProps> = ({ items, onSectionChange }) => {
  return (
    <Stack gap="density-sm" direction="row">
      {items.map((item) => {
        return (
          <Button key={item.id} kind="tertiary" onClick={() => onSectionChange?.(item.id)}>
            <Flex align="center" gap="density-lg">
              {item.icon}
              <Text kind="label/bold/md">{item.label}</Text>
            </Flex>
          </Button>
        );
      })}
    </Stack>
  );
};
