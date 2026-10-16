// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';

export type TestCaseVerdict = 'passed' | 'failed' | 'error' | 'not-scored';

type VerdictInput = Pick<
  EvaluationSessionResponse,
  'status' | 'evaluator_scores' | 'failed_evaluators'
>;

// Until evaluators carry their own pass threshold (ASTD-832), only a perfect score passes.
const PASSING_SCORE = 1;

export const getTestCaseVerdict = ({
  status,
  evaluator_scores,
  failed_evaluators,
}: VerdictInput): TestCaseVerdict => {
  if (status === 'error' || (failed_evaluators?.length ?? 0) > 0) {
    return 'error';
  }
  const scores = Object.values(evaluator_scores ?? {}).filter(
    (score): score is number => typeof score === 'number' && Number.isFinite(score)
  );
  if (scores.length === 0) {
    return 'not-scored';
  }
  if (scores.every((score) => score === PASSING_SCORE)) {
    return 'passed';
  }
  return 'failed';
};
