// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ADVANCED, BEGINNER } from '@studio/components/StartOptions/levels';
import type {
  OptimizationStrategyId,
  OptimizationStrategyOption,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import { FileUp, ListChecks } from 'lucide-react';

/** The guided form asks the least of the user, so the page opens on it. */
export const DEFAULT_STRATEGY: OptimizationStrategyId = 'form';

/** Rendered in order. The tile list scrolls, so a new strategy never pushes Continue off screen. */
export const STRATEGY_OPTIONS: OptimizationStrategyOption[] = [
  {
    id: 'form',
    title: 'Parameter sweep',
    description:
      'Answer three questions — what to tune for, what to score against, how many trials — and we build the study for you.',
    icon: ListChecks,
    tag: BEGINNER,
    enabled: true,
  },
  {
    id: 'upload',
    title: 'Upload a config',
    description:
      'Bring your own optimize bundle: the config plus the dataset and any other files it references.',
    icon: FileUp,
    tag: ADVANCED,
    enabled: true,
  },
];
