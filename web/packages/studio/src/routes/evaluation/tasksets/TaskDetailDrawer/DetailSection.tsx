// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface DetailSectionProps {
  heading: string;
  children: ReactNode;
}

export const DetailSection: FC<DetailSectionProps> = ({ heading, children }) => (
  <Stack gap="density-sm">
    <Text kind="label/bold/md">{heading}</Text>
    {children}
  </Stack>
);
