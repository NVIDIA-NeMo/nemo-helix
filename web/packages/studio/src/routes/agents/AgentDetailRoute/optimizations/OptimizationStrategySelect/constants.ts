// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOptionTag } from '@studio/components/StartOptions/types';
import type { OptimizationStrategyOption } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import { FileUp, Route, SlidersHorizontal, Sparkles } from 'lucide-react';

/** Marks a strategy the platform has but Studio cannot start yet. */
const COMING_SOON: StartOptionTag = { label: 'Coming soon', color: 'gray', kind: 'outline' };

/** Rendered in order. Picking a tile goes straight to that strategy. */
export const STRATEGY_OPTIONS: OptimizationStrategyOption[] = [
  {
    id: 'upload',
    title: 'Upload a config',
    description:
      'Bring your own optimize bundle, from your computer or a fileset: the config plus the dataset and any other files it references.',
    icon: FileUp,
    enabled: true,
  },
  {
    id: 'hyperparameter',
    title: 'Hyper-parameter optimization',
    description:
      'Answer three questions — what to tune for, what to score against, how many trials — and we build the study for you.',
    icon: SlidersHorizontal,
    enabled: true,
  },
  {
    id: 'skill',
    title: 'Skill optimization',
    description:
      'Rewrite the agent’s skills from where its evaluation runs fail, and keep the edits that improve the score.',
    icon: Sparkles,
    tag: COMING_SOON,
    enabled: false,
  },
  {
    id: 'routing',
    title: 'Model Routing (Switchyard)',
    description:
      'Send each request to the smallest model that can handle it, cutting cost without losing quality.',
    icon: Route,
    tag: COMING_SOON,
    enabled: false,
  },
];
