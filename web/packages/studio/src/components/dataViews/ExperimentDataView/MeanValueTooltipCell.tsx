// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getTextWithCount } from '@nemo/common/src/utils/formatters';
import { Badge, Flex, Text, Tooltip } from '@nvidia/foundations-react-core';
import { tooltipClassName } from '@studio/styles/common';
import { type FC, type ReactNode } from 'react';

const aggregateMetricTooltip = (
  label: string,
  count: number | null | undefined,
  runCount: number | null | undefined,
  countsMissingAsZero: boolean,
  failedCount: number
): string => {
  // `count` is the number of test cases the mean is taken over; `runCount` is the total attempts. The
  // rollup is test-case-weighted: each test case is averaged over its attempts first, then averaged
  // across test cases (so a test case run k times counts once, not k times).
  const testCases = count ?? 0;
  const attempts = runCount ?? 0;
  // When each test case was attempted once, the test-case-weighted mean is just the plain per-attempt
  // mean, so keep it simple. Otherwise note that each test case's repeats are averaged first.
  const base =
    testCases === attempts
      ? `Mean ${label} over ${getTextWithCount('test case', testCases)}.`
      : `Mean ${label} over ${getTextWithCount('test case', testCases)} — each test case averaged over its attempts.`;
  // Scores use a fixed denominator: a run that produced no score counts as 0, so it's included above
  // rather than dropped. Measurement metrics (cost, latency) instead omit missing values.
  const zeroNote = countsMissingAsZero ? ` Runs with no score count as 0.` : '';
  const failedNote =
    failedCount > 0
      ? ` ${getTextWithCount('run', failedCount)} failed to score and count as 0.`
      : '';
  return `${base}${zeroNote}${failedNote}`;
};

interface MeanValueTooltipCellProps {
  label: string;
  count: number | null | undefined;
  runCount: number | null | undefined;
  /** When true, note that errored runs with no score are counted as 0 (score metrics only). */
  readonly countsMissingAsZero?: boolean;
  /** Runs whose evaluator recorded a FAILED result; shown beside the mean and explained in the tooltip. */
  readonly failedCount?: number;
  children: ReactNode;
}

export const MeanValueTooltipCell: FC<MeanValueTooltipCellProps> = ({
  label,
  count,
  runCount,
  countsMissingAsZero = false,
  failedCount = 0,
  children,
}) => {
  if (count == null) {
    return <Text>{children}</Text>;
  }
  return (
    <Tooltip
      slotContent={
        <Text kind="body/regular/sm">
          {aggregateMetricTooltip(label, count, runCount, countsMissingAsZero, failedCount)}
        </Text>
      }
      className={tooltipClassName}
      side="bottom"
    >
      <Flex align="center" gap="density-xs" className="cursor-default">
        <Text className="border-b border-dotted border-brand">{children}</Text>
        {failedCount > 0 && (
          <Badge kind="outline" color="red">
            {failedCount} failed
          </Badge>
        )}
      </Flex>
    </Tooltip>
  );
};
