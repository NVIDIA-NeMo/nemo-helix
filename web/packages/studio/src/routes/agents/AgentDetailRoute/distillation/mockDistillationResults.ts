// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** `original`, `base`, or a candidate's name. */
export type ArmId = string;

export const ORIGINAL_ARM = 'original';
export const BASE_ARM = 'base';

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
  /** Which candidate this is, in order, for picking its colour. */
  candidateIndex?: number;
}

export const HELD_OUT_TASKS = 24;

const FIXED_COLOR_CLASSES: Record<string, string> = {
  [ORIGINAL_ARM]: 'bg-(--color-gray-400)',
  [BASE_ARM]: 'bg-(--color-blue-500)',
};
const FIXED_COLORS: Record<string, string> = {
  [ORIGINAL_ARM]: 'var(--color-gray-400)',
  [BASE_ARM]: 'var(--color-blue-500)',
};
const CANDIDATE_COLOR_CLASSES = ['bg-(--color-purple-500)', 'bg-(--color-teal-500)', 'bg-(--color-purple-900)', 'bg-(--color-gray-600)'];
const CANDIDATE_COLORS = ['var(--color-purple-500)', 'var(--color-teal-500)', 'var(--color-purple-900)', 'var(--color-gray-600)'];

/** One neutral color per model; green, yellow and red are kept for better or worse. */
export const armColorClass = (arm: MockArmResult): string =>
  FIXED_COLOR_CLASSES[arm.id] ??
  CANDIDATE_COLOR_CLASSES[(arm.candidateIndex ?? 0) % CANDIDATE_COLOR_CLASSES.length];

export const armColor = (arm: MockArmResult): string =>
  FIXED_COLORS[arm.id] ?? CANDIDATE_COLORS[(arm.candidateIndex ?? 0) % CANDIDATE_COLORS.length];
