// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getInsightsListAnalysisRunsQueryKey,
  insightsGetAnalysisRun,
  insightsListAnalysisRuns,
} from '@nemo/sdk/generated/insights/insights-analysis-runs';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { analysisJobStatus } from '@studio/api/insightsAnalysis';
import { useQuery } from '@tanstack/react-query';

const LOOKBACK_RUNS = 10;

/**
 * Start time of the agent's most recent completed analysis run, looking past newer runs that
 * failed or were cancelled. Null when none of the recent runs completed.
 */
export const useLastCompletedAnalysisRun = (workspace: string, agent: string) => {
  const params = { agent, page_size: LOOKBACK_RUNS, sort: '-created_at' };
  return useQuery({
    queryKey: [...getInsightsListAnalysisRunsQueryKey(workspace, params), 'last-completed'],
    queryFn: async ({ signal }) => {
      const { data: runs } = await insightsListAnalysisRuns(workspace, params, signal);
      for (const { name } of runs) {
        if (!name) continue;
        const { run, job } = await insightsGetAnalysisRun(workspace, name, signal);
        if (analysisJobStatus(job) === HelixJobStatus.completed) return run.created_at;
      }
      return null;
    },
  });
};
