// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import type { StudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import { HelixJobListSortField } from '@nemo/sdk/generated/platform/schema';
import {
  buildFilter,
  isJobRunning,
  isRecentJob,
} from '@studio/routes/WorkspaceDashboardHomeRoute/recentJobsUtils';
import { keepPreviousData } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';

const RECENT_WINDOW_MS = 48 * 60 * 60 * 1000;
const JOBS_FETCH_SIZE = 100;
// Used when nothing is running or the last poll failed. It picks up jobs started elsewhere and
// recovers from a jobs service restart without polling the full list every 5s.
const SLOW_POLLING_INTERVAL_MS = 30 * 1000;
const CUTOFF_BUCKET_MS = 60 * 60 * 1000;

const getFinishedSince = () =>
  new Date(
    Math.floor((Date.now() - RECENT_WINDOW_MS) / CUTOFF_BUCKET_MS) * CUTOFF_BUCKET_MS
  ).toISOString();

export const useRecentJobs = (workspace: string, dataViewState: StudioDataViewState) => {
  // Fixed at mount and rounded to the hour so polls and remounts reuse the query key.
  const [finishedSince] = useState(getFinishedSince);

  // The API filters status/updated_at after paging, so server pages come back short. Fetch one
  // large page and paginate client-side until it filters first.
  const {
    data: jobsData,
    dataUpdatedAt,
    isLoading,
    isPlaceholderData,
    error,
  } = useJobsListJobs(
    workspace,
    {
      page: 1,
      page_size: JOBS_FETCH_SIZE,
      sort: HelixJobListSortField['-created_at'],
      filter: buildFilter(finishedSince),
    },
    {
      query: {
        placeholderData: keepPreviousData,
        staleTime: 0,
        refetchOnWindowFocus: true,
        refetchInterval: (query) =>
          query.state.status !== 'error' &&
          query.state.data?.data.some((job) => isJobRunning(job.status))
            ? JOB_POLLING_INTERVAL_MS
            : SLOW_POLLING_INTERVAL_MS,
      },
    }
  );

  const recentJobs = useMemo(() => {
    const finishedSinceMs = dataUpdatedAt - RECENT_WINDOW_MS;
    return (jobsData?.data ?? []).filter((job) => isRecentJob(job, finishedSinceMs));
  }, [jobsData, dataUpdatedAt]);

  // Jobs age out between polls, so step back when the current page runs past the end.
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const lastPageIndex = Math.max(0, Math.ceil(recentJobs.length / pageSize) - 1);
  const safePageIndex = Math.min(pageIndex, lastPageIndex);
  // Wait for real rows: the empty loading list would discard a linked page.
  const hasFreshJobs = jobsData !== undefined && !isPlaceholderData;
  const setPagination = dataViewState.pagination.set;
  useEffect(() => {
    if (hasFreshJobs && pageIndex > lastPageIndex) {
      setPagination((prev) => ({ ...prev, pageIndex: lastPageIndex }));
    }
  }, [hasFreshJobs, pageIndex, lastPageIndex, setPagination]);
  const pageRows = useMemo(
    () => recentJobs.slice(safePageIndex * pageSize, safePageIndex * pageSize + pageSize),
    [recentJobs, safePageIndex, pageSize]
  );

  return {
    pageRows,
    totalCount: recentJobs.length,
    isLoading: isLoading || isPlaceholderData,
    // A failed poll keeps the last rows; only an initial load failure has nothing to show.
    error: jobsData === undefined ? error : null,
  };
};
