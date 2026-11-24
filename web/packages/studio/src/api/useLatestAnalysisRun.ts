// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { parseISOWithUTCFallback } from '@nemo/common/src/components/RelativeTime/util';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { useAgentsListExecuteJobs } from '@nemo/sdk/generated/agents/agents';
import {
  type AgentsListExecuteJobsParams,
  type ExecuteJobsListFilter,
  type ExecuteJobsPage,
  ExecuteJobsSortField,
} from '@nemo/sdk/generated/agents/schema';
import {
  getCompassListInsightsQueryKey,
  compassListInsights,
} from '@nemo/sdk/generated/compass/compass-insights';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema/HelixJobStatus';
import { analysisJobStatus, COMPASS_AGENT_TYPE } from '@studio/api/insightsAnalysis';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

export interface LatestAnalysisRun {
  name: string;
  requestedAt: string;
  status?: HelixJobStatus;
}

// Picks up runs started outside this page, such as by the scheduler or a trace import.
export const LIST_POLL_MS = 15_000;
const NEW_FINDINGS_PAGE_SIZE = 100;

const IN_FLIGHT_STATUSES: HelixJobStatus[] = [
  HelixJobStatus.created,
  HelixJobStatus.queued,
  HelixJobStatus.pending,
  HelixJobStatus.active,
];

export const latestAnalysisRunParams = (agent: string): AgentsListExecuteJobsParams => ({
  page_size: 1,
  sort: ExecuteJobsSortField['-created_at'],
  filter: jsonFilter<ExecuteJobsListFilter>({
    'spec.plugin_managed_agent_type': { $eq: COMPASS_AGENT_TYPE },
    'spec.subject': { $eq: agent },
  }),
});

const isInFlight = (status?: HelixJobStatus): boolean =>
  !!status && IN_FLIGHT_STATUSES.includes(status);

/** Polls fast while the latest run is in flight, and slowly otherwise to notice new runs. */
export const runPollInterval = (
  page: ExecuteJobsPage | undefined,
  fetchFailureCount: number
): number => {
  if (fetchFailureCount > 0) {
    return LIST_POLL_MS;
  }
  return isInFlight(analysisJobStatus(page?.data[0])) ? JOB_POLLING_INTERVAL_MS : LIST_POLL_MS;
};

const countInsightsSince = async (
  workspace: string,
  agent: string,
  since: string
): Promise<number | undefined> => {
  try {
    const { data } = await compassListInsights(workspace, {
      agent,
      page_size: NEW_FINDINGS_PAGE_SIZE,
      sort: '-created_at',
    });
    const start = parseISOWithUTCFallback(since).getTime();
    return data.filter(({ created_at }) => parseISOWithUTCFallback(created_at).getTime() >= start)
      .length;
  } catch {
    return undefined;
  }
};

const completionMessage = (added = 0) =>
  added > 0
    ? `Analysis complete: ${added} new ${added === 1 ? 'finding' : 'findings'}.`
    : 'Analysis complete.';

/** The agent's most recent analysis run, polled while in flight; toasts when a watched run ends. */
export const useLatestAnalysisRun = (
  workspace: string,
  agent: string | undefined
): { latestRun?: LatestAnalysisRun; isActive: boolean; blocksNewRun: boolean } => {
  const queryClient = useQueryClient();
  const toast = useToast();
  const watchedName = useRef<string | null>(null);

  const { data: page } = useAgentsListExecuteJobs(workspace, latestAnalysisRunParams(agent ?? ''), {
    query: {
      enabled: !!agent,
      refetchInterval: ({ state }) => runPollInterval(state.data, state.fetchFailureCount),
    },
  });
  const job = page?.data[0];
  const name = job?.name;
  const status = analysisJobStatus(job);
  const isActive = isInFlight(status);
  const requestedAt = job?.created_at;

  useEffect(() => {
    if (!agent || !name || !status || !requestedAt) return;

    if (isActive) {
      watchedName.current = name;
      return;
    }
    if (watchedName.current !== name) return;
    watchedName.current = null;

    if (status === HelixJobStatus.completed) {
      void (async () => {
        await queryClient.invalidateQueries({
          queryKey: getCompassListInsightsQueryKey(workspace),
        });
        toast.success(completionMessage(await countInsightsSince(workspace, agent, requestedAt)));
      })();
    } else if (status === HelixJobStatus.error) {
      toast.error(`Analysis run "${name}" failed. Open the run's job to read its logs.`);
    }
  }, [agent, isActive, name, queryClient, requestedAt, status, toast, workspace]);

  const latestRun: LatestAnalysisRun | undefined =
    name && requestedAt ? { name, requestedAt, status } : undefined;

  return { latestRun, isActive, blocksNewRun: isActive };
};
