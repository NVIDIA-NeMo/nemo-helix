// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentEvalResultSummary } from '@nemo/sdk/generated/evals/schema';
import { Badge, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface EvalMetricCoverageTableProps {
  summary: AgentEvalResultSummary;
}

interface CoverageRow {
  name: string;
  total: number;
  scored: number;
  failed: number;
  missing: number;
}

/** One row per metric output, named the way the aggregate scores table names them. */
const coverageRows = (summary: AgentEvalResultSummary): CoverageRow[] =>
  Object.entries(summary.metric_coverage ?? {}).flatMap(([metric, outputs]) =>
    Object.entries(outputs ?? {}).map(([output, coverage]) => ({
      name: output === metric ? metric : `${metric}.${output}`,
      total: coverage.total ?? 0,
      scored: coverage.scored ?? 0,
      failed: coverage.failed ?? 0,
      missing: coverage.missing ?? 0,
    }))
  );

const countColor = (
  count: number,
  kind: 'scored' | 'failed' | 'missing'
): 'green' | 'red' | 'yellow' | 'gray' => {
  if (count === 0) return 'gray';
  if (kind === 'scored') return 'green';
  return kind === 'failed' ? 'red' : 'yellow';
};

/** Per-output coverage from the run summary: how many trials a metric actually scored, how many
 *  failed, and how many it never emitted. This is what separates a low mean from low coverage. */
export const EvalMetricCoverageTable: FC<EvalMetricCoverageTableProps> = ({ summary }) => {
  const rows = coverageRows(summary);
  if (rows.length === 0) return null;
  return (
    <Stack gap="density-sm" data-testid="eval-metric-coverage">
      <Text kind="body/semibold/md">Coverage</Text>
      <Text kind="body/regular/sm" className="text-secondary">
        {summary.trial_count} {summary.trial_count === 1 ? 'trial' : 'trials'} across{' '}
        {summary.task_count} {summary.task_count === 1 ? 'task' : 'tasks'}; {summary.error_count}{' '}
        {summary.error_count === 1 ? 'trial reported an error' : 'trials reported errors'}.
      </Text>
      <Stack gap="density-xs">
        {rows.map((row) => (
          <Flex key={row.name} align="center" justify="between" gap="density-md" className="w-full">
            <Text kind="body/semibold/sm" className="truncate" title={row.name}>
              {row.name}
            </Text>
            <Flex gap="density-xs" className="shrink-0">
              <Badge kind="solid" color={countColor(row.scored, 'scored')}>
                {row.scored}/{row.total} scored
              </Badge>
              <Badge kind="solid" color={countColor(row.failed, 'failed')}>
                {row.failed} failed
              </Badge>
              <Badge kind="solid" color={countColor(row.missing, 'missing')}>
                {row.missing} missing
              </Badge>
            </Flex>
          </Flex>
        ))}
      </Stack>
    </Stack>
  );
};
