// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile, type StatTileDiagnosticProps } from '@nemo/common/src/components/StatTile/index';
import { Grid, Panel, Skeleton, Stack } from '@nvidia/foundations-react-core';
import type { TraceStatisticsSummary } from '@studio/components/AgentTraceStatistics/types';
import { formatLatencyMs, formatTokens } from '@studio/components/AgentTraceStatistics/utils';
import { formatCost } from '@studio/util/intakeTelemetry';
import { type FC } from 'react';

interface Props {
  /** `null` while the first rollup is still in flight — the skeletons stand in for it. */
  summary: TraceStatisticsSummary | null;
  isPending?: boolean;
}

const EMPTY_SUMMARY: TraceStatisticsSummary = {
  totalTraces: 0,
  avgLatencyMs: 0,
  avgTokensPerRun: 0,
  avgCostUsd: 0,
};

export const TraceStatisticsTiles: FC<Props> = ({ summary, isPending }) => {
  const { totalTraces, avgLatencyMs, avgTokensPerRun, avgCostUsd } = summary ?? EMPTY_SUMMARY;
  const tiles: StatTileDiagnosticProps[] = [
    { label: 'Total traces', value: formatTokens(totalTraces) },
    { label: 'Avg latency', value: formatLatencyMs(avgLatencyMs), trailingLabel: 'per run' },
    {
      label: 'Avg token count',
      value: formatTokens(avgTokensPerRun),
      trailingLabel: 'per run',
    },
    { label: 'Avg cost', value: formatCost(avgCostUsd), trailingLabel: 'per run' },
  ];

  return (
    <Grid cols={{ base: 1, md: 2, lg: 4 }} gap="density-lg">
      {tiles.map((tile) =>
        isPending ? (
          <Panel key={tile.label} className="w-full bg-surface-raised">
            <Stack gap="density-xxs">
              <Skeleton className="h-[21px] w-24" />
              <Skeleton className="h-8 w-20" />
            </Stack>
          </Panel>
        ) : (
          <StatTile key={tile.label} {...tile} variant="metric" className="bg-surface-raised" />
        )
      )}
    </Grid>
  );
};
