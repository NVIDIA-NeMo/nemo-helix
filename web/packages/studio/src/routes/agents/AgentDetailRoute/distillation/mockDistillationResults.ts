// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type ArmId = 'original' | 'base' | 'distilled';

export interface MockArmResult {
  id: ArmId;
  title: string;
  model: string;
  score: number;
  inputTokensPerTask: number;
  outputTokensPerTask: number;
  medianLatencySec: number;
  estimatedCostPer1kTasks: number | null;
  evalJobName: string;
}

export const HELD_OUT_TASKS = 24;

/** One neutral color per model; green, yellow and red are kept for better or worse. */
export const ARM_COLOR_CLASSES: Record<ArmId, string> = {
  original: 'bg-(--color-gray-400)',
  base: 'bg-(--color-blue-500)',
  distilled: 'bg-(--color-purple-500)',
};

export const ARM_COLORS: Record<ArmId, string> = {
  original: 'var(--color-gray-400)',
  base: 'var(--color-blue-500)',
  distilled: 'var(--color-purple-500)',
};
