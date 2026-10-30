// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isServerError } from '@nemo/common/src/api/common/utils';
import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import type { StudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getJobsListJobsQueryKey, useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import { HelixJobListSortField } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';
import {
  buildFilter,
  isJobRunning,
} from '@studio/routes/WorkspaceDashboardHomeRoute/recentJobsUtils';
import { useEffect, useState } from 'react';

const RECENT_WINDOW_MS = 48 * 60 * 60 * 1000;
const SLOW_POLLING_INTERVAL_MS = 30 * 1000;
const CUTOFF_BUCKET_MS = 5 * 60 * 1000;

const getFinishedSince = (now: number) =>
  new Date(
    Math.floor((now - RECENT_WINDOW_MS) / CUTOFF_BUCKET_MS) * CUTOFF_BUCKET_MS
  ).toISOString();

export const useRecentJobs = (workspace: string, dataViewState: StudioDataViewState) => {
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  // Rounded so polls within a bucket reuse the query key.
  const [finishedSince, setFinishedSince] = useState(() => getFinishedSince(Date.now()));
  const [lastTotal, setLastTotal] = useState<{ workspace: string; total: number }>();
  const [lastError, setLastError] = useState<Error>();
  const [jobsPath] = getJobsListJobsQueryKey(workspace);

  const {
    data: jobsData,
    dataUpdatedAt,
    isFetched,
    isFetching,
    isPending,
    isPlaceholderData,
    error,
    refetch,
  } = useJobsListJobs(
    workspace,
    {
      page: pageIndex + 1,
      page_size: pageSize,
      sort: HelixJobListSortField['-created_at'],
      filter: buildFilter(finishedSince),
    },
    {
      query: {
        // Never show another workspace's rows.
        placeholderData: (previousData, previousQuery) =>
          previousQuery?.queryKey[0] === jobsPath ? previousData : undefined,
        staleTime: 0,
        retry: (failureCount, error) =>
          isServerError(error) && failureCount < DEFAULT_QUERY_RETRY_COUNT,
        refetchOnWindowFocus: true,
        refetchInterval: (query) =>
          query.state.status !== 'error' &&
          query.state.data?.data.some((job) => isJobRunning(job.status))
            ? JOB_POLLING_INTERVAL_LONG
            : SLOW_POLLING_INTERVAL_MS,
      },
    }
  );

  useEffect(() => {
    if (dataUpdatedAt) {
      setFinishedSince(getFinishedSince(dataUpdatedAt));
    }
  }, [dataUpdatedAt]);

  // A refetch clears the error of a query with no data, so keep it to show meanwhile.
  useEffect(() => {
    if (error) {
      setLastError(error);
    }
  }, [error]);
  // This key's last fetch failed if it has been fetched but still has no rows of its own.
  const isLoadFailed = isFetched && (jobsData === undefined || isPlaceholderData);

  const fetchedTotal = jobsData?.pagination?.total_results;
  useEffect(() => {
    if (fetchedTotal !== undefined && !isPlaceholderData) {
      setLastTotal({ workspace, total: fetchedTotal });
    }
  }, [fetchedTotal, isPlaceholderData, workspace]);
  const totalCount =
    fetchedTotal ?? (lastTotal?.workspace === workspace ? lastTotal.total : undefined) ?? 0;

  const lastPageIndex = Math.max(0, Math.ceil(totalCount / pageSize) - 1);
  const isBeyondLastPage = jobsData !== undefined && pageIndex > lastPageIndex;
  // Cached totals can be stale, so wait for the refetch.
  const isPastEnd = isBeyondLastPage && !isPlaceholderData && !isFetching;
  const setPagination = dataViewState.pagination.set;
  useEffect(() => {
    if (isPastEnd) {
      setPagination((prev) => ({ ...prev, pageIndex: lastPageIndex }));
    }
  }, [isPastEnd, lastPageIndex, setPagination]);

  // Only a page change, not a cutoff change, shows as loading.
  const isChangingPage =
    isPlaceholderData &&
    (jobsData?.pagination?.page !== pageIndex + 1 || jobsData?.pagination?.page_size !== pageSize);

  return {
    pageRows: jobsData?.data ?? [],
    totalCount,
    // `isPending`, not `isLoading`: a first load paused offline isn't fetching but has no rows.
    isLoading: isPending || isChangingPage || isBeyondLastPage,
    error: isLoadFailed ? (error ?? lastError ?? null) : null,
    refetch,
  };
};
