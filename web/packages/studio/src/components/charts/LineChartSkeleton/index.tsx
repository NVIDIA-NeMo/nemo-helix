// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Flex, Stack, Skeleton } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

export const LineChartSkeleton: FC = () => {
  return (
    <Stack align="center" className="w-full h-full" gap="density-sm">
      {/* Skeleton for chart title */}
      <Skeleton className="self-start w-3/5 h-[30px]" />

      {/* Skeleton for the line chart area */}
      <Block className="relative w-full h-full">
        <Skeleton className="w-full h-full" />
      </Block>

      {/* Skeleton for chart's x-axis labels */}
      <Flex justify="between" className="w-full">
        {[...Array(6)].map((_, index) => (
          <Skeleton key={index} className="w-1/6 h-[20px]" />
        ))}
      </Flex>
    </Stack>
  );
};
