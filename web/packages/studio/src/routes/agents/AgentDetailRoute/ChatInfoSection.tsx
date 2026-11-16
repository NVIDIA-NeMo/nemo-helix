// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface ChatInfoSectionProps {
  title: string;
  children: ReactNode;
}

/** A titled section of the Chat tab's information column, in the Overview tab's style. */
export const ChatInfoSection: FC<ChatInfoSectionProps> = ({ title, children }) => (
  <Stack role="region" aria-label={title} gap="density-md">
    <Text kind="title/sm">{title}</Text>
    {children}
  </Stack>
);
