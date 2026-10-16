// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatEvaluatorScore, snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import type { EvaluationSessionResponse } from '@nemo/sdk/generated/platform/schema';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { type FC, type ReactNode } from 'react';

const SummaryItem: FC<{ label: string; children: ReactNode }> = ({ label, children }) => (
  <Stack gap="density-xs" className="min-w-0">
    <Text kind="body/regular/sm" className="truncate text-secondary">
      {label}
    </Text>
    {children}
  </Stack>
);

export const TestCaseScores: FC<{ run: EvaluationSessionResponse }> = ({ run }) => {
  const scores = Object.entries(run.evaluator_scores ?? {});
  const scoredNames = new Set(scores.map(([name]) => name));
  const unscoredFailures = (run.failed_evaluators ?? []).filter((name) => !scoredNames.has(name));

  if (scores.length === 0 && unscoredFailures.length === 0) {
    return null;
  }
  return (
    <Flex
      align="start"
      gap="density-2xl"
      className="flex-wrap rounded-sm border border-base bg-surface-raised p-density-lg"
      data-testid="test-case-scores"
    >
      {scores.map(([name, score]) => (
        <SummaryItem key={name} label={snakeCaseToTitleCase(name)}>
          <Text kind="title/xs">{formatEvaluatorScore(score)}</Text>
        </SummaryItem>
      ))}
      {unscoredFailures.map((name) => (
        <SummaryItem key={name} label={snakeCaseToTitleCase(name)}>
          <Badge kind="outline" color="red">
            Failed
          </Badge>
        </SummaryItem>
      ))}
    </Flex>
  );
};
