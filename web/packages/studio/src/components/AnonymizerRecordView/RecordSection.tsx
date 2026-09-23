// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface RecordSectionProps {
  readonly heading: string;
  readonly className?: string;
  readonly children: ReactNode;
}

/** Shared by the record view and its skeleton so the two stay aligned. */
export const RecordSection: FC<RecordSectionProps> = ({ heading, className, children }) => (
  <Stack className={className} gap="density-md">
    <Text className="text-secondary" kind="label/regular/md">
      {heading}
    </Text>
    {children}
  </Stack>
);
