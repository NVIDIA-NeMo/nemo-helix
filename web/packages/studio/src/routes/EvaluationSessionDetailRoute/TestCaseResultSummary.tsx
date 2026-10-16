// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { MetricScoreChip } from '@studio/components/evaluation/MetricScoreChip';
import { getTestCaseVerdict } from '@studio/components/evaluation/testCaseVerdict';
import { TestCaseVerdictBadge } from '@studio/components/evaluation/TestCaseVerdictBadge';
import { type FC } from 'react';

const FAILED_TO_SCORE = 'failed';

export const TestCaseResultSummary: FC<{ run: EvaluationSessionResponse }> = ({ run }) => {
  const scores = Object.entries(run.evaluator_scores ?? {});
  const scoredNames = new Set(scores.map(([name]) => name));
  const unscoredFailures = (run.failed_evaluators ?? []).filter((name) => !scoredNames.has(name));

  return (
    <Flex
      align="start"
      gap="density-2xl"
      className="flex-wrap rounded-sm border border-base bg-surface-raised p-density-lg"
      data-testid="test-case-result-summary"
    >
      <Stack gap="density-xs">
        <Text kind="body/regular/sm" className="text-secondary">
          Result
        </Text>
        <TestCaseVerdictBadge verdict={getTestCaseVerdict(run)} />
      </Stack>
      {scores.map(([name, score]) => (
        <MetricScoreChip key={name} label={snakeCaseToTitleCase(name)} value={score} />
      ))}
      {unscoredFailures.map((name) => (
        <MetricScoreChip key={name} label={snakeCaseToTitleCase(name)} value={FAILED_TO_SCORE} />
      ))}
    </Flex>
  );
};
