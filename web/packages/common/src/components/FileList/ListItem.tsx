// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@nvidia/foundations-react-core';
import { InfoCircle } from '@nvidia/react-gui-icons/icons';
import { FC, ReactNode } from 'react';

interface ListItemProps {
  value: string;
  startIconSlot?: ReactNode;
  outlined?: boolean;
  endIconSlot?: ReactNode;
  error?: boolean;
}

export const ListItem: FC<ListItemProps> = ({
  value,
  startIconSlot,
  endIconSlot,
  outlined,
  error,
}) => {
  return (
    <Flex
      gap="density-sm"
      align="center"
      className={`px-2 h-full min-h-10 ${outlined ? 'border border-interaction-base rounded-md' : ''}`}
    >
      {startIconSlot || (error && <InfoCircle className="text-feedback-danger" />)}
      <Text kind="body/regular/md" className={`mr-auto ${error ? 'text-feedback-danger' : ''}`}>
        {value}
      </Text>
      {endIconSlot && <Flex gap="density-lg">{endIconSlot}</Flex>}
    </Flex>
  );
};
