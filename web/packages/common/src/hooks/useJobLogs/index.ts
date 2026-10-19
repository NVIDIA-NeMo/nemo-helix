// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobsPageJobLogsQueryKey, jobsPageJobLogs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobLog,
  HelixJobLogPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import {
  hashKey,
  useQuery,
  useQueryClient,
  type QueryObserverResult,
  type RefetchOptions,
} from '@tanstack/react-query';
import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';

import { LOGS_MAX_FETCH_ITERATIONS, LOGS_MAX_PAGES, LOGS_PAGE_SIZE } from '../../constants';
import { CJobTerminalStatuses } from '../../constants/query';
import type { LogLoadProgress } from '../../utils/logs';
import { getJobRefetchInterval } from '../../utils/query';

// After a job goes terminal, refetchInterval stops polling — but OTLP log
// shipping can still be in flight, so the final lines would be lost. Refetch a
// few times post-terminal to capture the tail. Bounded and self-clearing.
const LOG_SETTLE_DELAYS_MS = [2_000, 6_000, 12_000];

// Progress lives in a module store keyed by query key rather than in component state,
// for two reasons. The log query key is shared — a viewer and a download button can
// observe the same job — and React Query runs the queryFn of whichever observer won
// the fetch, so per-instance state freezes on the loser's last count. And the hook is
// not remounted when the :jobName route param changes, so per-instance state would
// paint the previous job's count for the frame before the new queryFn resets it.
const progressByQueryKey = new Map<string, LogLoadProgress | null>();
const progressListeners = new Map<string, Set<() => void>>();

function reportProgress(queryKeyHash: string, progress: LogLoadProgress | null): void {
  const listeners = progressListeners.get(queryKeyHash);
  // Nothing mounted is watching this job — don't retain an entry for it. Keeping the
  // store bounded to live subscribers is what makes unsubscribe a safe place to evict.
  if (!listeners) return;
  progressByQueryKey.set(queryKeyHash, progress);
  listeners.forEach((listener) => listener());
}

function subscribeToProgress(queryKeyHash: string, listener: () => void): () => void {
  const listeners = progressListeners.get(queryKeyHash) ?? new Set<() => void>();
  progressListeners.set(queryKeyHash, listeners);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0) {
      progressListeners.delete(queryKeyHash);
      progressByQueryKey.delete(queryKeyHash);
    }
  };
}

export interface UseJobLogsOptions {
  workspace: string;
  name: string;
  enabled?: boolean;
  jobStatus?: HelixJobStatus;
  pageSize?: number;
  /** Max pages of logs to retain in memory. Defaults to LOGS_MAX_PAGES.
   *  Set to Infinity for download scenarios where all logs are needed. */
  maxPages?: number;
}

interface JobLogsQueryData {
  logs: HelixJobLog[];
  total: number;
}

export interface UseJobLogsResult {
  data: HelixJobLog[];
  isLoading: boolean;
  error: Error | null;
  total: number;
  /** Advances as the cursor walk pages through the log, `null` until the first page
   *  resolves and again whenever a new walk starts. Meant for the initial load, which
   *  blocks on every page; refetches already have data on screen. */
  loadProgress: LogLoadProgress | null;
  refetch: (options?: RefetchOptions) => Promise<QueryObserverResult<JobLogsQueryData>>;
}

export function getJobLogsQueryKey(
  workspace: string,
  name: string
): [...ReturnType<typeof getJobsPageJobLogsQueryKey>, 'all'] {
  return [...getJobsPageJobLogsQueryKey(workspace, name), 'all'];
}

function getPageQueryKey(workspace: string, name: string, cursor: string | undefined) {
  return [...getJobsPageJobLogsQueryKey(workspace, name), 'page', cursor ?? 'initial'];
}

export const useJobLogs = ({
  workspace,
  name,
  enabled,
  jobStatus,
  pageSize = LOGS_PAGE_SIZE,
  maxPages = LOGS_MAX_PAGES,
}: UseJobLogsOptions): UseJobLogsResult => {
  const queryClient = useQueryClient();
  const queryKey = getJobLogsQueryKey(workspace, name);
  const queryKeyHash = hashKey(queryKey);
  const maxRetainedLogs = maxPages * pageSize;

  // Subscribed before useQuery so this effect registers ahead of the one that kicks
  // off the fetch — otherwise the walk's first report would land with no listener.
  const loadProgress = useSyncExternalStore(
    useCallback(
      (listener: () => void) => subscribeToProgress(queryKeyHash, listener),
      [queryKeyHash]
    ),
    useCallback(() => progressByQueryKey.get(queryKeyHash) ?? null, [queryKeyHash])
  );

  const query = useQuery<JobLogsQueryData>({
    queryKey,
    queryFn: async ({ signal }) => {
      let allLogs: HelixJobLog[] = [];
      let cursor: string | undefined;
      let total = 0;
      // Counted separately from allLogs, which the retention cap trims back to the
      // tail — otherwise progress would stall at maxRetainedLogs on exactly the long
      // logs this is here to narrate.
      let fetched = 0;

      reportProgress(queryKeyHash, null);

      for (let i = 0; i < LOGS_MAX_FETCH_ITERATIONS; i++) {
        if (signal.aborted) break;

        const pageCursor = cursor;
        const pageKey = getPageQueryKey(workspace, name, pageCursor);
        const cached = queryClient.getQueryData<HelixJobLogPage>(pageKey);
        const isCachedFullPage = cached !== undefined && cached.data.length >= pageSize;

        const page = await queryClient.fetchQuery({
          queryKey: pageKey,
          queryFn: ({ signal }) =>
            jobsPageJobLogs(workspace, name, { limit: pageSize, page_cursor: pageCursor }, signal),
          staleTime: isCachedFullPage ? Infinity : 0,
        });

        allLogs.push(...page.data);
        total = page.total;
        fetched += page.data.length;

        if (maxRetainedLogs !== Infinity && allLogs.length > maxRetainedLogs) {
          allLogs = allLogs.slice(-maxRetainedLogs);
        }

        if (!signal.aborted) reportProgress(queryKeyHash, { loaded: fetched, total });

        if (!page.next_page || page.data.length === 0) break;
        cursor = page.next_page;
      }

      return { logs: allLogs, total };
    },
    enabled: enabled ?? !!(workspace && name),
    refetchInterval: () => getJobRefetchInterval(jobStatus),
  });

  const isTerminal = !!jobStatus && CJobTerminalStatuses.includes(jobStatus);
  const { refetch } = query;
  // Only settle-burst when the job COMPLETES while mounted (a non-terminal ->
  // terminal transition we actually observed) — not when mounting into an
  // already-terminal job, whose initial fetch already has the full log. This
  // also stops the burst re-firing on remount, e.g. re-expanding a collapsed
  // log panel on a finished job.
  const sawActiveRef = useRef(false);
  useEffect(() => {
    if (jobStatus && !isTerminal) sawActiveRef.current = true;
    if (!isTerminal || !sawActiveRef.current) return;
    const timers = LOG_SETTLE_DELAYS_MS.map((ms) => setTimeout(() => void refetch(), ms));
    return () => timers.forEach(clearTimeout);
  }, [jobStatus, isTerminal, refetch]);

  return {
    data: query.data?.logs ?? [],
    isLoading: query.isLoading,
    error: query.error,
    total: query.data?.total ?? 0,
    loadProgress,
    refetch: query.refetch,
  };
};
