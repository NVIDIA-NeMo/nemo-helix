// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@studio/util/functions';

/** Key the evaluator job writes its run rollup under in the job's `status_details`. */
export const RUN_OUTCOME_STATUS_DETAILS_KEY = 'evaluation';

export type EvalRunUnit = 'trials' | 'rows';

/** The evaluator job's own verdict on a run: how many units it produced, how many reported an
 *  error, how many ended with a usable score, and whether that adds up to a failed run. */
export interface EvalRunOutcome {
  readonly unit: EvalRunUnit;
  readonly total: number;
  readonly errored: number;
  readonly scored: number;
  readonly failed: boolean;
  readonly message: string;
}

const isCount = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0;

const isUnit = (value: unknown): value is EvalRunUnit => value === 'trials' || value === 'rows';

/** The run outcome recorded in a job's `status_details`, or null when the job predates it, is not an
 *  evaluator job, or carries a malformed block. */
export const evalRunOutcomeOf = (statusDetails: unknown): EvalRunOutcome | null => {
  if (!isPlainObject(statusDetails)) return null;
  const block = statusDetails[RUN_OUTCOME_STATUS_DETAILS_KEY];
  if (!isPlainObject(block)) return null;
  const { unit, total, errored, scored, failed, message } = block;
  if (!isUnit(unit) || !isCount(total) || !isCount(errored) || !isCount(scored)) return null;
  if (typeof failed !== 'boolean' || typeof message !== 'string') return null;
  return { unit, total, errored, scored, failed, message };
};

export type EvalRunOutcomeSeverity = 'error' | 'warning';

/** Whether the outcome is worth interrupting the reader for. A failed run is an error; a completed
 *  run with any unscored or errored unit is a warning; a clean run is neither. */
export const evalRunOutcomeSeverity = (outcome: EvalRunOutcome): EvalRunOutcomeSeverity | null => {
  if (outcome.failed) return 'error';
  if (outcome.errored > 0 || outcome.scored < outcome.total) return 'warning';
  return null;
};

/** Compact "27 of 28 scored" text for list cells. */
export const evalRunOutcomeLabel = (outcome: EvalRunOutcome): string =>
  `${outcome.scored} of ${outcome.total} scored`;
