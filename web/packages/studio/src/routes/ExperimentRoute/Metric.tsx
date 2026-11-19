// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import type { FC, ReactNode } from 'react';

interface MetricProps {
  title: string;
  value: ReactNode;
  icon?: ReactNode;
}

export const Metric: FC<MetricProps> = ({ title, value, icon }) => (
  <div className="flex flex-col items-center gap-1.5">
    <Text kind="label/semibold/sm" className="text-tertiary uppercase">
      {title}
    </Text>
    <div className="flex items-center gap-1">
      {icon && <span className="size-4 shrink-0 [&>svg]:size-4">{icon}</span>}
      <Text kind="body/regular/xl">{value}</Text>
    </div>
  </div>
);
