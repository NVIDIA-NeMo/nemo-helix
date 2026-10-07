// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption } from '@studio/components/StartOptions/types';

/**
 * Every way into a new optimization. A new id needs a tile in `STRATEGY_OPTIONS` and a handler in
 * `OptimizationsTab`; the tab will not compile until it has one.
 */
export type OptimizationStrategyId = 'form' | 'upload';

export type OptimizationStrategyOption = StartOption<OptimizationStrategyId>;

export interface OptimizationStrategySelectProps {
  agentName?: string;
  /** Returns to the studies table. */
  onBack: () => void;
  /** Fired when the user confirms a strategy via the Continue footer. */
  onContinue: (strategy: OptimizationStrategyId) => void;
}
