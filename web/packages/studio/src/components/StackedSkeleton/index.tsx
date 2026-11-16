// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Skeleton, type SkeletonProps } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface Props extends SkeletonProps {
  count: number;
  height?: number;
}

export const StackedSkeleton: FC<Props> = ({ count, height, ...skeletonProps }) => {
  return (
    <>
      {Array.from({ length: Math.max(count, 1) }).map((_, index) => (
        // eslint-disable-next-line no-restricted-syntax
        <Skeleton key={index} style={{ height }} {...skeletonProps} />
      ))}
    </>
  );
};
