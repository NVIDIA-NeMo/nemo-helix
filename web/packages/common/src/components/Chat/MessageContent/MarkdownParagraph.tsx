// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { FC, PropsWithChildren } from 'react';

export const MarkdownParagraph: FC<PropsWithChildren> = ({ children }) => (
  <Text asChild kind="body/regular/md">
    <p className="mb-density-md text-sm leading-6 last:mb-0">{children}</p>
  </Text>
);
