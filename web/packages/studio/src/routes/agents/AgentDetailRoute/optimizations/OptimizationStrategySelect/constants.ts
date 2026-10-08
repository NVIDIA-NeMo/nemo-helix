// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OptimizationStrategyOption } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import { FileUp, ListChecks } from 'lucide-react';

/** Rendered in order. Picking a tile goes straight to that strategy. */
export const STRATEGY_OPTIONS: OptimizationStrategyOption[] = [
  {
    id: 'form',
    title: 'Parameter sweep',
    description:
      'Answer three questions — what to tune for, what to score against, how many trials — and we build the study for you.',
    icon: ListChecks,
    enabled: true,
  },
  {
    id: 'upload',
    title: 'Upload a config',
    description:
      'Bring your own optimize bundle: the config plus the dataset and any other files it references.',
    icon: FileUp,
    enabled: true,
  },
];
