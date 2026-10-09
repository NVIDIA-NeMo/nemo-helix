// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseISOWithUTCFallback } from '@nemo/common/src/components/RelativeTime/util';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
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
import type { AnalysisRunResponse } from '@nemo/sdk/generated/insights/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { analysisJobStatus } from '@studio/api/insightsAnalysis';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

export interface LatestAnalysisRun {
  name: string;
  requestedAt: string;
  /** False when the run was recorded but its job never landed. */
  submitted: boolean;
  status?: HelixJobStatus;
}

// Picks up runs started outside this page, such as by the scheduler or a trace import.
export const LIST_POLL_MS = 15_000;
// The scheduler records a run just before submitting its job.
const UNSUBMITTED_GRACE_MS = 60_000;
const NEW_FINDINGS_PAGE_SIZE = 100;

const IN_FLIGHT_STATUSES: HelixJobStatus[] = [
  HelixJobStatus.created,
  HelixJobStatus.pending,
  HelixJobStatus.active,
];

const runStatusPollInterval = (data: AnalysisRunResponse): number | false => {
  if (!data.job) {
    const age = Date.now() - parseISOWithUTCFallback(data.run.created_at).getTime();
    return age < UNSUBMITTED_GRACE_MS ? JOB_POLLING_INTERVAL_MS : false;
  }
  const status = analysisJobStatus(data.job);
  return status ? getJobRefetchInterval(status) : false;
};

export const runPollInterval = (
  data: AnalysisRunResponse | undefined,
  fetchFailureCount: number
): number | false => {
  if (!data) return false;
  const interval = runStatusPollInterval(data);
  return interval !== false && fetchFailureCount > 0 ? LIST_POLL_MS : interval;
};

const countInsightsSince = async (
  workspace: string,
  agent: string,
  since: string
): Promise<number | undefined> => {
  try {
    const { data } = await insightsListInsights(workspace, {
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

  const { data: runs } = useInsightsListAnalysisRuns(
    workspace,
    { agent, page_size: 1, sort: '-created_at' },
    { query: { enabled: !!agent, refetchInterval: LIST_POLL_MS } }
  );
  const name = runs?.data[0]?.name;

  const { data: detail, isPending: detailPending } = useInsightsGetAnalysisRun(
    workspace,
    name ?? '',
    {
      query: {
        enabled: !!name,
        refetchInterval: ({ state }) => runPollInterval(state.data, state.fetchFailureCount),
      },
    }
  );

  const current = detail && detail.run.name === name ? detail : undefined;
  const status = analysisJobStatus(current?.job);
  const isActive = !!status && IN_FLIGHT_STATUSES.includes(status);
  const requestedAt = current?.run.created_at;

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
          queryKey: getInsightsListInsightsQueryKey(workspace),
        });
        toast.success(completionMessage(await countInsightsSince(workspace, agent, requestedAt)));
      })();
    } else if (status === HelixJobStatus.error) {
      toast.error(`Analysis run "${name}" failed. Open the run's job to read its logs.`);
    }
  }, [agent, isActive, name, queryClient, requestedAt, status, toast, workspace]);

  const latestRun: LatestAnalysisRun | undefined =
    current && name
      ? { name, requestedAt: current.run.created_at, submitted: !!current.job, status }
      : undefined;

  return { latestRun, isActive, blocksNewRun: isActive || (!!name && detailPending) };
};
