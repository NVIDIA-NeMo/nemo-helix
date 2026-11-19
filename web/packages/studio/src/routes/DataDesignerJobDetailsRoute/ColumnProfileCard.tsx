// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Card, Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { CategoricalHistogramChart } from '@studio/routes/DataDesignerJobDetailsRoute/CategoricalHistogramChart';
import {
  formatPercent,
  formatStatCount,
  formatStatDecimal,
  getCategoricalHistogram,
  getColumnTypeLabel,
  getNumericalDistribution,
  getPercentNull,
  getPercentUnique,
  isLLMColumnStatistics,
  isValidationColumnStatistics,
  type ColumnStatistics,
} from '@studio/routes/DataDesignerJobDetailsRoute/datasetProfilerTypes';
import type { FC } from 'react';

interface StatProps {
  label: string;
  value: string;
}

const Stat: FC<StatProps> = ({ label, value }) => (
  <Stack gap="density-xxs" className="min-w-0">
    <Text kind="body/regular/xs" className="text-muted uppercase tracking-wide">
      {label}
    </Text>
    <Text kind="body/regular/md" className="truncate">
      {value}
    </Text>
  </Stack>
);

interface ColumnProfileCardProps {
  stats: ColumnStatistics;
}

/**
 * Builds the column-specific detail body: a bar chart for categorical sampler
 * distributions, a numeric summary for numerical samplers, token usage for LLM
 * columns, and valid-record counts for validation columns. Returns `null` when
 * a column has no extra detail (e.g. a uuid sampler), so the caller can skip the
 * divider rather than render an empty section.
 */
const renderColumnDetail = (stats: ColumnStatistics): React.ReactNode => {
  const histogram = getCategoricalHistogram(stats);
  if (histogram) {
    return <CategoricalHistogramChart histogram={histogram} />;
  }

  const numerical = getNumericalDistribution(stats);
  if (numerical) {
    return (
      <Flex gap="density-xl" className="flex-wrap">
        <Stat label="Min" value={formatStatDecimal(numerical.min)} />
        <Stat label="Max" value={formatStatDecimal(numerical.max)} />
        <Stat label="Mean" value={formatStatDecimal(numerical.mean)} />
        <Stat label="Median" value={formatStatDecimal(numerical.median)} />
        <Stat label="Std dev" value={formatStatDecimal(numerical.stddev)} />
      </Flex>
    );
  }

  if (isLLMColumnStatistics(stats)) {
    return (
      <Flex gap="density-xl" className="flex-wrap">
        <Stat label="Input tokens (avg)" value={formatStatDecimal(stats.input_tokens_mean)} />
        <Stat label="Output tokens (avg)" value={formatStatDecimal(stats.output_tokens_mean)} />
      </Flex>
    );
  }

  if (isValidationColumnStatistics(stats)) {
    return (
      <Flex gap="density-xl" className="flex-wrap">
        <Stat label="Valid records" value={formatStatCount(stats.num_valid_records)} />
      </Flex>
    );
  }

  return null;
};

/** A single column's profile rendered as a self-contained card for the grid. */
export const ColumnProfileCard: FC<ColumnProfileCardProps> = ({ stats }) => {
  const detail = renderColumnDetail(stats);

  return (
    <Card className="h-full">
      <Stack gap="density-md" className="h-full">
        <Stack gap="density-md">
          <Stack gap="density-xxs">
            <Flex justify="between" align="center" gap="density-sm">
              <Text kind="body/bold/md" className="truncate font-mono">
                {stats.column_name}
              </Text>
              <Badge kind="outline">{getColumnTypeLabel(stats)}</Badge>
            </Flex>
            <Text kind="body/regular/xs" className="text-muted font-mono">
              {stats.simple_dtype}
            </Text>
          </Stack>

          <Flex gap="density-xl" className="flex-wrap">
            <Stat label="Records" value={formatStatCount(stats.num_records)} />
            <Stat label="Null %" value={formatPercent(getPercentNull(stats))} />
            <Stat label="Unique %" value={formatPercent(getPercentUnique(stats))} />
          </Flex>
          {detail && <Divider />}
        </Stack>

        {detail}
      </Stack>
    </Card>
  );
};
