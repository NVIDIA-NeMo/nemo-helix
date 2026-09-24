// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Card, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import type { MonitorSummary } from '@studio/routes/agents/AgentMonitorRoute/utils';
import { FC } from 'react';

interface Props {
  summary: MonitorSummary;
}

interface Tile {
  label: string;
  value: string;
  hint?: string;
}

const formatNumber = (value: number): string => value.toLocaleString();

export const SummaryCards: FC<Props> = ({ summary }) => {
  const agentsHint =
    summary.uniqueAgents > 0
      ? `${summary.uniqueAgents} agent${summary.uniqueAgents === 1 ? '' : 's'}`
      : undefined;

  const tiles: Tile[] = [
    {
      label: 'Total runs',
      value: formatNumber(summary.totalRuns),
      hint: agentsHint,
    },
    {
      label: 'Avg prompt tokens',
      value: formatNumber(summary.avgPromptTokens),
      hint: 'per run',
    },
    {
      label: 'Avg completion tokens',
      value: formatNumber(summary.avgCompletionTokens),
      hint: 'per run',
    },
    {
      label: 'Tool calls',
      value: formatNumber(summary.totalToolCalls),
      hint: summary.topModelCount
        ? `Top model: ${summary.topModel} (${summary.topModelCount})`
        : undefined,
    },
  ];

  return (
    <Grid cols={{ base: 1, md: 2, lg: 4 }} gap="density-xl">
      {tiles.map((tile) => (
        <Card key={tile.label}>
          <Stack gap="density-sm" padding="density-xl">
            <Text className="text-secondary" kind="body/regular/sm">
              {tile.label}
            </Text>
            <Text kind="body/bold/2xl">{tile.value}</Text>
            {tile.hint ? (
              <Text kind="body/regular/sm" className="truncate text-secondary" title={tile.hint}>
                {tile.hint}
              </Text>
            ) : null}
          </Stack>
        </Card>
      ))}
    </Grid>
  );
};
