// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption } from '@studio/components/StartOptions/types';

/**
 * Every way into a new optimization that Studio can start. A new id needs a tile in
 * `STRATEGY_OPTIONS` and a handler in `OptimizationsTab`; the tab will not compile until it has one.
 */
export type OptimizationStrategyId = 'hyperparameter' | 'routing' | 'upload';

/** Strategies shown in the picker but disabled until Studio can start them. */
export type PlannedOptimizationStrategyId = 'skill';

export type OptimizationStrategyOption = StartOption<
  OptimizationStrategyId | PlannedOptimizationStrategyId
>;

export interface OptimizationStrategySelectProps {
  agentName?: string;
  /** Returns to the studies table. */
  onBack: () => void;
  /** Fired as soon as the user picks a strategy. Disabled tiles never fire it. */
  onSelect: (strategy: OptimizationStrategyId) => void;
}
