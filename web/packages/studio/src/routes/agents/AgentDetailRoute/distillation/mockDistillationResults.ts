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
  estimatedCostPer1kTasks: number;
  evalJobName: string;
}

export interface TokenPrice {
  inputPerMillion: number;
  outputPerMillion: number;
}

export const HELD_OUT_TASKS = 24;

/** Illustrative prices. A fine-tuned adapter is served on its base model, so it costs the same. */
export const ORIGINAL_PRICE: TokenPrice = { inputPerMillion: 0.88, outputPerMillion: 0.88 };
export const STUDENT_PRICE: TokenPrice = { inputPerMillion: 0.15, outputPerMillion: 0.6 };

export const ORIGINAL_MODEL = 'default/llama-3-3-70b-instruct';

export const estimateCostPer1kTasks = (
  inputTokensPerTask: number,
  outputTokensPerTask: number,
  price: TokenPrice
): number =>
  (1000 * (inputTokensPerTask * price.inputPerMillion + outputTokensPerTask * price.outputPerMillion)) /
  1_000_000;

// TODO(api): read each arm's aggregate scores from its agent-eval-results record
// (GET .../agent-eval-results/{evalJobName}), with tokens and latency from the trial measurements.
// Cost stays computed client-side with estimateCostPer1kTasks until the backend prices runs.
export const buildMockResults = (distillationName: string, studentModel: string): MockArmResult[] => [
  {
    id: 'original',
    title: 'Baseline',
    model: ORIGINAL_MODEL,
    score: 0.88,
    inputTokensPerTask: 3200,
    outputTokensPerTask: 450,
    medianLatencySec: 9.8,
    estimatedCostPer1kTasks: estimateCostPer1kTasks(3200, 450, ORIGINAL_PRICE),
    evalJobName: `${distillationName}-eval-original`,
  },
  {
    id: 'base',
    title: 'Model swap',
    model: studentModel,
    score: 0.54,
    inputTokensPerTask: 3400,
    outputTokensPerTask: 700,
    medianLatencySec: 3.1,
    estimatedCostPer1kTasks: estimateCostPer1kTasks(3400, 700, STUDENT_PRICE),
    evalJobName: `${distillationName}-eval-base`,
  },
  {
    id: 'distilled',
    title: 'After distillation',
    model: `${studentModel} + adapter`,
    score: 0.85,
    inputTokensPerTask: 3150,
    outputTokensPerTask: 430,
    medianLatencySec: 3.4,
    estimatedCostPer1kTasks: estimateCostPer1kTasks(3150, 430, STUDENT_PRICE),
    evalJobName: `${distillationName}-eval-distilled`,
  },
];

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

export const ARM_PRICES: Record<ArmId, TokenPrice> = {
  original: ORIGINAL_PRICE,
  base: STUDENT_PRICE,
  distilled: STUDENT_PRICE,
};
