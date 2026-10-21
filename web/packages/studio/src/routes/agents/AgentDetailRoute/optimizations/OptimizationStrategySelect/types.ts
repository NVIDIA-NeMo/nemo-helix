// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { OptimizationStrategy } from '@nemo/sdk/generated/agent-optimization/schema/OptimizationStrategy';

export interface OptimizationStrategySelectProps {
  agentName?: string;
  /** Returns to the studies table. */
  onBack: () => void;
  /** Fired as soon as the user picks an installed strategy. */
  onSelect: (strategy: OptimizationStrategy) => void;
  /** Fired when the user picks the upload tile, to bring their own `config-tuning` bundle. */
  onUploadConfig: () => void;
}
