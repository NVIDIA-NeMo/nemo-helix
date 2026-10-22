// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Text } from '@nvidia/foundations-react-core';
import { Equal, Subtract, Add } from '@nvidia/react-gui-icons/icons';
import type { ComparisonMetricDelta } from '@studio/components/dataViews/EvalComparisonTable/types';
import { formatScore } from '@studio/components/evaluation/utils';
import type { FC } from 'react';

export interface ComparisonDeltaCellProps {
  readonly delta: ComparisonMetricDelta;
  /** When false, a lower score is the improvement (latency, cost, error rate). */
  readonly higherIsBetter?: boolean;
}

/** A metric value for one run, annotated with its signed change against the baseline run.
 * Green means the run improved on the baseline, red means it regressed. */
export const ComparisonDeltaCell: FC<ComparisonDeltaCellProps> = ({
  delta,
  higherIsBetter = true,
}) => {
  const { difference, value } = delta;

  return (
    <Flex align="center" gap="density-md" data-testid="eval-delta-cell">
      <Text className="min-w-[3.5rem] tabular-nums" kind="body/regular/md">
        {formatScore(value)}
      </Text>
      {difference !== null && (
        <DeltaBadge difference={difference} higherIsBetter={higherIsBetter} />
      )}
    </Flex>
  );
};

const DeltaBadge: FC<{ difference: number; higherIsBetter: boolean }> = ({
  difference,
  higherIsBetter,
}) => {
  if (difference === 0) {
    return (
      <Badge
        color="gray"
        kind="solid"
        aria-label="No change versus baseline"
        data-delta="unchanged"
      >
        <Equal aria-hidden size="12" />
      </Badge>
    );
  }

  const improved = higherIsBetter ? difference > 0 : difference < 0;
  const DeltaIcon = difference > 0 ? Add : Subtract;

  return (
    <Badge
      color={improved ? 'green' : 'red'}
      kind="solid"
      aria-label={`${improved ? 'Improved' : 'Regressed'} by ${Math.abs(difference).toFixed(3)} versus baseline`}
      data-delta={improved ? 'improved' : 'regressed'}
    >
      <DeltaIcon aria-hidden size="12" />
      <span className="tabular-nums">{Math.abs(difference).toFixed(3)}</span>
    </Badge>
  );
};
