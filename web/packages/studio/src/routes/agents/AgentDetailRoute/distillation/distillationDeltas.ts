// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MockArmResult } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';

export type DeltaTone = 'success' | 'warning' | 'error';

export interface Delta {
  text: string;
  tone: DeltaTone;
}

/** A worse score by this many points, or a worse cost or latency by this percent, reads as red. */
const SEVERE_SCORE_DROP = 0.1;
const SEVERE_PERCENT_WORSENING = 25;

const arrow = (change: number): string => (change > 0 ? '▲' : '▼');

export const percentChange = (value: number, reference: number): number =>
  Math.round(((value - reference) / reference) * 100);

/** Green when better; yellow for a small regression and red for a severe one. */
const toDelta = (
  change: number,
  amount: string,
  higherIsBetter: boolean,
  severeAt: number
): Delta | undefined => {
  if (change === 0) return undefined;
  const text = `${arrow(change)} ${amount} vs baseline`;
  if (higherIsBetter ? change > 0 : change < 0) return { text, tone: 'success' };
  return { text, tone: Math.abs(change) >= severeAt ? 'error' : 'warning' };
};

export const scoreDelta = (arm: MockArmResult, baseline: MockArmResult): Delta | undefined => {
  const change = arm.score - baseline.score;
  return toDelta(change, Math.abs(change).toFixed(2), true, SEVERE_SCORE_DROP);
};

export const costDelta = (arm: MockArmResult, baseline: MockArmResult): Delta | undefined => {
  if (arm.estimatedCostPer1kTasks === null || baseline.estimatedCostPer1kTasks === null) return undefined;
  const change = percentChange(arm.estimatedCostPer1kTasks, baseline.estimatedCostPer1kTasks);
  return toDelta(change, `${Math.abs(change)}%`, false, SEVERE_PERCENT_WORSENING);
};

export const latencyDelta = (arm: MockArmResult, baseline: MockArmResult): Delta | undefined => {
  const change = percentChange(arm.medianLatencySec, baseline.medianLatencySec);
  return toDelta(change, `${Math.abs(change)}%`, false, SEVERE_PERCENT_WORSENING);
};
