// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Skeleton, Stack } from '@nvidia/foundations-react-core';

export const HistoryPanelSkeleton = () => (
  <Stack gap="density-sm" padding="density-md">
    <Skeleton className="h-16 w-full" />
    <Skeleton className="h-16 w-full" />
    <Skeleton className="h-16 w-full" />
  </Stack>
);

export const SkillsPanelSkeleton = () => (
  <Stack gap="density-sm" padding="density-md">
    <Skeleton className="h-24 w-full" />
    <Skeleton className="h-24 w-full" />
    <Skeleton className="h-24 w-full" />
  </Stack>
);
