// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Stack, Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

export interface DetailSectionProps {
  heading: string;
  description?: string;
  children: ReactNode;
}

export const DetailSection: FC<DetailSectionProps> = ({ heading, description, children }) => {
  return (
    <Stack asChild gap="density-lg" className="w-full">
      <section>
        <Stack gap="density-xs">
          <Text kind="label/bold/lg" asChild>
            <h2>{heading}</h2>
          </Text>
          {description && (
            <Text kind="body/regular/sm" className="text-secondary">
              {description}
            </Text>
          )}
        </Stack>
        {children}
      </section>
    </Stack>
  );
};
