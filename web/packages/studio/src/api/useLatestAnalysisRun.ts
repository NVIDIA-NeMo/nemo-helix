// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import {
  useInsightsGetAnalysisRun,
  useInsightsListAnalysisRuns,
} from '@nemo/sdk/generated/insights/insights-analysis-runs';
import {
  getInsightsListInsightsQueryKey,
  insightsListInsights,
} from '@nemo/sdk/generated/insights/insights-insights';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { analysisJobStatus } from '@studio/api/insightsAnalysis';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

export interface LatestAnalysisRun {
  name: string;
  startedAt: string;
  /** False when the run was recorded but its job never landed. */
  submitted: boolean;
  status?: HelixJobStatus;
}

interface WatchedRun {
  name: string;
  insightsBefore?: number;
}

const countInsights = async (workspace: string, agent: string): Promise<number | undefined> => {
  try {
    const { pagination } = await insightsListInsights(workspace, { agent, page_size: 1 });
    return pagination?.total_results;
  } catch {
    return undefined;
  }
};

const completionMessage = (added: number) =>
  added > 0
    ? `Analysis complete: ${added} new ${added === 1 ? 'finding' : 'findings'}.`
    : 'Analysis complete.';

/**
 * The agent's most recent analysis run with its job state, polled until the job is terminal.
 * A run seen in flight toasts when it finishes and refreshes the insights list.
 */
export const useLatestAnalysisRun = (
  workspace: string,
  agent: string | undefined
): { latestRun?: LatestAnalysisRun; isActive: boolean } => {
  const queryClient = useQueryClient();
  const toast = useToast();
  const watched = useRef<WatchedRun | null>(null);

  const { data: runs } = useInsightsListAnalysisRuns(
    workspace,
    { agent, page_size: 1, sort: '-created_at' },
    { query: { enabled: !!agent } }
  );
  const name = runs?.data[0]?.name;

  const { data: detail } = useInsightsGetAnalysisRun(workspace, name ?? '', {
    query: {
      enabled: !!name,
      refetchInterval: ({ state }) =>
        state.data && !state.data.job
          ? false
          : getJobRefetchInterval(analysisJobStatus(state.data?.job)),
    },
  });

  const status = analysisJobStatus(detail?.job);
  const isActive = !!status && !CJobTerminalStatuses.includes(status);

  useEffect(() => {
    if (!agent || !name || !status) return;

    if (isActive) {
      if (watched.current?.name === name) return;
      const run: WatchedRun = { name };
      watched.current = run;
      void countInsights(workspace, agent).then((count) => {
        run.insightsBefore = count;
      });
      return;
    }

    if (watched.current?.name !== name) return;
    const { insightsBefore } = watched.current;
    watched.current = null;

    if (status === HelixJobStatus.completed) {
      void (async () => {
        await queryClient.invalidateQueries({
          queryKey: getInsightsListInsightsQueryKey(workspace),
        });
        const insightsAfter = await countInsights(workspace, agent);
        const added =
          insightsBefore !== undefined && insightsAfter !== undefined
            ? insightsAfter - insightsBefore
            : 0;
        toast.success(completionMessage(added));
      })();
    } else if (status === HelixJobStatus.error) {
      toast.error(`Analysis run "${name}" failed. Open the run's job to read its logs.`);
    }
  }, [agent, isActive, name, queryClient, status, toast, workspace]);

  const latestRun: LatestAnalysisRun | undefined =
    detail && name && detail.run.name === name
      ? { name, startedAt: detail.run.created_at, submitted: !!detail.job, status }
      : undefined;

  return { latestRun, isActive };
};
