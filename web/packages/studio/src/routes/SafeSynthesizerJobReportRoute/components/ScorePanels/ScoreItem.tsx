// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@nvidia/foundations-react-core';
import { CheckCircle, Cancel } from '@nvidia/react-gui-icons/icons';
import { FC } from 'react';

interface ScoreItemProps {
  success: boolean;
  value: string;
}
export const ScoreItem: FC<ScoreItemProps> = ({ success, value }) => {
  return (
    <Flex align="center" gap="density-md">
      {success ? <CheckCircle className="text-brand" /> : <Cancel />}
      <Text>{value}</Text>
    </Flex>
  );
};
