// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type ArmId,
  HELD_OUT_TASKS,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';

/** 1 passed, 0.5 partial, 0 failed. */
export type TaskOutcome = 0 | 0.5 | 1;

export interface MockTask {
  number: number;
  outcomes: Record<ArmId, TaskOutcome>;
}

export interface WinTieLoss {
  wins: number;
  ties: number;
  losses: number;
}

const FAILED: Record<ArmId, number[]> = {
  original: [5, 12, 19],
  base: [1, 2, 5, 7, 9, 12, 14, 16, 19, 20, 22],
  distilled: [5, 12, 21],
};
const PARTIAL: Record<ArmId, number[]> = { original: [], base: [], distilled: [19] };

const outcomeOf = (arm: ArmId, index: number): TaskOutcome =>
  FAILED[arm].includes(index) ? 0 : PARTIAL[arm].includes(index) ? 0.5 : 1;

// TODO(api): per-task outcomes come from each evaluation's agent-eval-results bundle, matched
// across runs by task id.
export const MOCK_TASKS: MockTask[] = Array.from({ length: HELD_OUT_TASKS }, (_, index) => {
  const outcomes = {
    original: outcomeOf('original', index),
    base: outcomeOf('base', index),
    distilled: outcomeOf('distilled', index),
  };
  return {
    number: index + 1,
    outcomes,
  };
});

/** How a challenger fares against the original, task by task. */
export const winTieLoss = (arm: ArmId): WinTieLoss =>
  MOCK_TASKS.reduce<WinTieLoss>(
    (total, task) => {
      const difference = task.outcomes[arm] - task.outcomes.original;
      return {
        wins: total.wins + (difference > 0 ? 1 : 0),
        ties: total.ties + (difference === 0 ? 1 : 0),
        losses: total.losses + (difference < 0 ? 1 : 0),
      };
    },
    { wins: 0, ties: 0, losses: 0 }
  );

export const outcomeCounts = (arm: ArmId): { passed: number; partial: number; failed: number } => ({
  passed: MOCK_TASKS.filter((task) => task.outcomes[arm] === 1).length,
  partial: MOCK_TASKS.filter((task) => task.outcomes[arm] === 0.5).length,
  failed: MOCK_TASKS.filter((task) => task.outcomes[arm] === 0).length,
});
