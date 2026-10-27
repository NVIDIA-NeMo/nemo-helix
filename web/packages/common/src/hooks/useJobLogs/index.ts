// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  LOGS_MAX_FETCH_ITERATIONS,
  LOGS_MAX_PAGE_SIZE,
  LOGS_TAIL_SIZE,
} from '@nemo/common/src/constants';
import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { isSameLogEntry, type LogPaging } from '@nemo/common/src/utils/logs';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { getJobsPageJobLogsQueryKey, jobsPageJobLogs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobLog,
  HelixJobLogPage,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import {
  useQuery,
  useQueryClient,
  type QueryObserverResult,
  type RefetchOptions,
} from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

// After a job goes terminal, refetchInterval stops polling — but OTLP log
// shipping can still be in flight, so the final lines would be lost. Refetch a
// few times post-terminal to capture the tail. Bounded and self-clearing.
const LOG_SETTLE_DELAYS_MS = [2_000, 6_000, 12_000];

export interface UseJobLogsOptions {
  workspace: string;
  name: string;
  enabled?: boolean;
  jobStatus?: HelixJobStatus;
}

interface JobLogsQueryData {
  logs: HelixJobLog[];
  total: number;
  /** Cursor for the lines just older than `logs[0]`; null once the start of the log is held. */
  olderCursor: string | null;
  /** Lines the last poll brought in, so the next one's window can reach back past as many. */
  growth: number;
}

export interface UseJobLogsResult {
  data: HelixJobLog[];
  isLoading: boolean;
  error: Error | null;
  total: number;
  paging: LogPaging;
  refetch: (options?: RefetchOptions) => Promise<QueryObserverResult<JobLogsQueryData>>;
}

export function getJobLogsQueryKey(
  workspace: string,
  name: string
): [...ReturnType<typeof getJobsPageJobLogsQueryKey>, 'all'] {
  return [...getJobsPageJobLogsQueryKey(workspace, name), 'all'];
}

function toQueryData(page: HelixJobLogPage): JobLogsQueryData {
  // The server sends null when there's no older page; the generated type and mocks use ''.
  return { logs: page.data, total: page.total, olderCursor: page.prev_page || null, growth: 0 };
}

/** Folds a fresh tail window into the lines already held, or returns null if it doesn't reach
 *  back to them. The window replaces everything from where it starts, so lines shipped late
 *  into that span land in place. */
function mergeTail(
  prev: JobLogsQueryData | undefined,
  page: HelixJobLogPage
): JobLogsQueryData | null {
  const [first] = page.data;
  if (!prev?.logs.length || !first) {
    return toQueryData(page);
  }

  // Any overlap starts within the last window's worth of held lines.
  const searchFrom = Math.max(0, prev.logs.length - page.data.length);
  for (let i = prev.logs.length - 1; i >= searchFrom; i--) {
    if (isSameLogEntry(prev.logs[i], first)) {
      return {
        logs: [...prev.logs.slice(0, i), ...page.data],
        total: page.total,
        olderCursor: prev.olderCursor,
        growth: Math.max(0, page.total - prev.total),
      };
    }
  }
  return null;
}

/** Every line of a job's log, oldest first, paged back from the tail at the server's max. */
export async function fetchAllJobLogs(workspace: string, name: string): Promise<HelixJobLog[]> {
  let page = await jobsPageJobLogs(workspace, name, { tail: LOGS_MAX_PAGE_SIZE });
  const pages = [page.data];
  for (let i = 0; page.prev_page && i < LOGS_MAX_FETCH_ITERATIONS; i++) {
    page = await jobsPageJobLogs(workspace, name, {
      page_cursor: page.prev_page,
      limit: LOGS_MAX_PAGE_SIZE,
    });
    pages.push(page.data);
  }
  return pages.reverse().flat();
}

export const useJobLogs = ({
  workspace,
  name,
  enabled,
  jobStatus,
}: UseJobLogsOptions): UseJobLogsResult => {
  const queryClient = useQueryClient();
  const queryKey = getJobLogsQueryKey(workspace, name);

  const query = useQuery<JobLogsQueryData>({
    queryKey,
    queryFn: async ({ signal }) => {
      // Reach back past as many lines as the last poll brought in, so a busy log still
      // overlaps the held lines in one request.
      const growth = queryClient.getQueryData<JobLogsQueryData>(queryKey)?.growth ?? 0;
      const tail = Math.min(LOGS_MAX_PAGE_SIZE, LOGS_TAIL_SIZE + growth);
      const page = await jobsPageJobLogs(workspace, name, { tail }, signal);
      // Read after each await, so older lines fetched while this was in flight are kept.
      const prev = queryClient.getQueryData<JobLogsQueryData>(queryKey);
      const merged = mergeTail(prev, page);
      // More lines arrived than the window holds: widen it back to the held lines, so older
      // pages the user loaded aren't dropped.
      const wider = Math.min(LOGS_MAX_PAGE_SIZE, page.total - (prev?.total ?? 0) + LOGS_TAIL_SIZE);
      if (merged || wider <= tail) {
        return merged ?? toQueryData(page);
      }
      const widerPage = await jobsPageJobLogs(workspace, name, { tail: wider }, signal);
      // Still no overlap (past the API's cap, or a new attempt): start over.
      const latest = queryClient.getQueryData<JobLogsQueryData>(queryKey);
      return mergeTail(latest, widerPage) ?? toQueryData(widerPage);
    },
    enabled: enabled ?? !!(workspace && name),
    refetchInterval: () => getJobRefetchInterval(jobStatus),
  });

  const [isFetchingOlderLogs, setIsFetchingOlderLogs] = useState(false);
  const isFetchingOlderRef = useRef(false);
  const fetchOlderLogs = useCallback(async () => {
    const key = getJobLogsQueryKey(workspace, name);
    const cursor = queryClient.getQueryData<JobLogsQueryData>(key)?.olderCursor;
    if (!cursor || isFetchingOlderRef.current) {
      return;
    }

    isFetchingOlderRef.current = true;
    setIsFetchingOlderLogs(true);
    try {
      const page = await jobsPageJobLogs(workspace, name, {
        page_cursor: cursor,
        limit: LOGS_MAX_PAGE_SIZE,
      });
      queryClient.setQueryData<JobLogsQueryData>(key, (latest) =>
        // A poll that started over since this was sent has moved on from this cursor.
        latest?.olderCursor === cursor
          ? {
              ...latest,
              logs: [...page.data, ...latest.logs],
              olderCursor: page.prev_page || null,
            }
          : latest
      );
    } finally {
      isFetchingOlderRef.current = false;
      setIsFetchingOlderLogs(false);
    }
  }, [queryClient, workspace, name]);

  const fetchAllLogs = useCallback(() => fetchAllJobLogs(workspace, name), [workspace, name]);

  const hasOlderLogs = !!query.data?.olderCursor;
  const paging = useMemo<LogPaging>(
    () => ({ hasOlderLogs, isFetchingOlderLogs, fetchOlderLogs, fetchAllLogs }),
    [hasOlderLogs, isFetchingOlderLogs, fetchOlderLogs, fetchAllLogs]
  );

  const isTerminal = !!jobStatus && CJobTerminalStatuses.includes(jobStatus);
  const { refetch } = query;
  // Only settle-burst when the job COMPLETES while mounted (a non-terminal ->
  // terminal transition we actually observed) — not when mounting into an
  // already-terminal job, whose initial fetch already has the full log. This
  // also stops the burst re-firing on remount, e.g. re-expanding a collapsed
  // log panel on a finished job.
  const sawActiveRef = useRef(false);
  useEffect(() => {
    if (jobStatus && !isTerminal) {
      sawActiveRef.current = true;
    }
    if (!isTerminal || !sawActiveRef.current) {
      return;
    }
    const timers = LOG_SETTLE_DELAYS_MS.map((ms) => setTimeout(() => void refetch(), ms));
    return () => timers.forEach(clearTimeout);
  }, [jobStatus, isTerminal, refetch]);

  return {
    data: query.data?.logs ?? [],
    isLoading: query.isLoading,
    error: query.error,
    total: query.data?.total ?? 0,
    paging,
    refetch: query.refetch,
  };
};
