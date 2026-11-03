// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import { useCompassGetAnalysisConfig } from '@nemo/sdk/generated/compass/compass-analysis-configs';
import { useCompassListInsights } from '@nemo/sdk/generated/compass/compass-insights';
import { Grid } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface InsightsMetricsProps {
  workspace: string;
  agent: string;
}

const statusOf = (error: unknown): number | undefined =>
  (error as { response?: { status?: number } } | null)?.response?.status;

/** Reads only the total from a one-row page; a failed read shows "—", never a false 0. */
const useInsightCount = (workspace: string, agent: string, status?: 'open') => {
  const { data, isLoading, isError } = useCompassListInsights(workspace, {
    agent,
    page_size: 1,
    ...(status ? { status } : {}),
  });
  return {
    value: isLoading || isError ? '—' : String(data?.pagination?.total_results ?? 0),
    isError,
  };
};

/** Headline numbers for an agent's insights: how many, how many still open, and whether it is on a schedule. */
export const InsightsMetrics: FC<InsightsMetricsProps> = ({ workspace, agent }) => {
  const total = useInsightCount(workspace, agent);
  const open = useInsightCount(workspace, agent, 'open');

  const {
    data: config,
    isLoading: isConfigLoading,
    error: configError,
  } = useCompassGetAnalysisConfig(workspace, agent, { query: { retry: false } });
  const notSetUp = statusOf(configError) === 404;
  const configUnreadable = !!configError && !notSetUp;

  return (
    <Grid cols={{ base: 1, md: 3 }} gap="density-xl" className="shrink-0">
      <StatTile
        variant="metric"
        label="Insights"
        value={total.value}
        status={total.isError ? 'error' : undefined}
      />
      <StatTile
        variant="metric"
        label="Open insights"
        value={open.value}
        status={open.isError ? 'error' : undefined}
      />
      <StatTile
        variant="metric"
        label="Periodic analysis"
        value={isConfigLoading || configUnreadable ? '—' : config?.enabled ? 'Enabled' : 'Off'}
        trailingLabel={notSetUp ? 'Not set up' : undefined}
        status={configUnreadable ? 'error' : undefined}
      />
    </Grid>
  );
};
