// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { ElapsedTime } from '@nemo/common/src/components/ElapsedTime';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { utcToLocalDate } from '@nemo/common/src/utils/date';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import {
  type HelixJobListSortField,
  type HelixJobResponse,
  type HelixJobsListFilter,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { Flex, Tag } from '@nvidia/foundations-react-core';
import {
  HIDDEN_JOB_SOURCES,
  JOB_SOURCE,
} from '@studio/components/dataViews/JobsDataView/constants';
import { SOURCE_DISPLAY } from '@studio/components/dataViews/JobsDataView/sourceDisplay';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { IN_PROGRESS_JOB_STATUSES } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { iconColorClass } from '@studio/routes/constants';
import { getNewCustomizationJobRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { Cog } from 'lucide-react';
import { type ComponentProps, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

const RECENT_WINDOW_MS = 48 * 60 * 60 * 1000;
const JOBS_FETCH_SIZE = 100;

const HIDDEN_SOURCES = CUSTOMIZER_ENABLED
  ? HIDDEN_JOB_SOURCES
  : [...HIDDEN_JOB_SOURCES, JOB_SOURCE.CUSTOMIZATION];

/**
 * Jobs that are still running, or that finished within the last 48 hours. Jobs carry no
 * completion timestamp, so `updated_at` stands in for when a terminal job finished.
 */
const buildFilter = (finishedSince: string): HelixJobsListFilter =>
  jsonFilter<HelixJobsListFilter>({
    $and: [
      { source: { $nin: HIDDEN_SOURCES } },
      {
        $or: [
          { status: { $in: IN_PROGRESS_JOB_STATUSES } },
          {
            $and: [
              { status: { $in: HelixJobTerminalStatuses } },
              { updated_at: { $gte: finishedSince } },
            ],
          },
        ],
      },
    ],
  });

/** `buildFilter`'s test for a later cutoff than the request's: in progress, or finished since. */
const isRecentJob = (job: HelixJobResponse, finishedSinceMs: number): boolean => {
  if (IN_PROGRESS_JOB_STATUSES.includes(job.status)) return true;
  const finishedAt = utcToLocalDate(job.updated_at);
  return finishedAt !== undefined && finishedAt.getTime() >= finishedSinceMs;
};

/** Still moving; a paused job's clock stops at the pause, like a finished job's. */
const isJobRunning = (status: HelixJobStatus): boolean =>
  IN_PROGRESS_JOB_STATUSES.includes(status) && status !== HelixJobStatus.paused;

export const RecentJobsTable = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const openRow = useRowNavigation();
  // Fixed at mount so the query key stays stable across polls; `recentJobs` keeps the window moving.
  const [finishedSince] = useState(() => new Date(Date.now() - RECENT_WINDOW_MS).toISOString());

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    defaultPageSize: 10,
    // Hidden, but still the default sort: newest first.
    columnVisibility: { created_at: false },
  });

  // The jobs API checks status and updated_at only after it pages (both live on the job's attempt),
  // so a server page of this filter comes back short and its total counts every job. Until the API
  // filters before paging, fetch the newest JOBS_FETCH_SIZE jobs at once and page them here.
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
      sort: getSortParam(dataViewState.sorting.state) as HelixJobListSortField,
      filter: buildFilter(finishedSince),
    },
    {
      query: {
        placeholderData: keepPreviousData,
        staleTime: 0,
        refetchOnWindowFocus: true,
        refetchInterval: (query) =>
          query.state.status === 'error' ? false : JOB_POLLING_INTERVAL_MS,
      },
    }
  );

  // The request's cutoff stays fixed at mount, so narrow its rows to each poll's 48-hour window.
  const recentJobs = useMemo(() => {
    const finishedSinceMs = dataUpdatedAt - RECENT_WINDOW_MS;
    return (jobsData?.data ?? []).filter((job) => isRecentJob(job, finishedSinceMs));
  }, [jobsData, dataUpdatedAt]);

  // StudioDataView renders exactly the rows it is handed, so apply the page here. Jobs age out of
  // the window between polls, so step back when the page in view runs past the end.
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const lastPageIndex = Math.max(0, Math.ceil(recentJobs.length / pageSize) - 1);
  const safePageIndex = Math.min(pageIndex, lastPageIndex);
  // Only once real rows are in: the empty list while loading would discard a linked page.
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

  const makeColumns: ComponentProps<typeof StudioDataView<HelixJobResponse>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor((original) => original?.name || '', {
      id: 'name',
      header: 'Name',
    }),
    accessor((original) => original?.source || '', {
      id: 'source',
      header: 'Source',
      meta: {
        title: (cell) => {
          const value = cell.getValue();
          return typeof value === 'string' && value
            ? (SOURCE_DISPLAY[value]?.label ?? value)
            : undefined;
        },
      },
      cell: ({ row }) => {
        const sourceValue = row.original.source;
        if (!sourceValue) return '-';
        const display = SOURCE_DISPLAY[sourceValue];
        return (
          <Tag kind="outline" color="gray" density="compact" readOnly>
            <Flex align="center" gap="density-sm">
              {display?.icon ?? <Cog className={iconColorClass} size={14} />}
              {display?.label ?? sourceValue}
            </Flex>
          </Tag>
        );
      },
    }),
    accessor((original) => original?.status || '', {
      id: 'status',
      header: 'Status',
      cell: ({ row }) => <StatusBadge status={row.original.status} />,
    }),
    accessor((original) => original?.created_at || '', {
      id: 'created_at',
      header: 'Created',
      enableSorting: true,
      cell: ({ row }) => <RelativeTime datetime={row.original.created_at ?? ''} />,
    }),
    accessor((original) => original?.created_at || '', {
      id: 'elapsed',
      header: 'Elapsed time',
      // The accessor value is only the start, which the automatic cell tooltip would show raw.
      meta: { title: false },
      // Keyed by job so a live count never carries over from another job's start.
      cell: ({ row }) => (
        <ElapsedTime
          key={row.original.id}
          start={row.original.created_at}
          end={row.original.updated_at}
          running={isJobRunning(row.original.status)}
        />
      ),
    }),
  ];

  if (error) {
    return <ErrorPanel errorMessage={getErrorMessage(error)} />;
  }

  return (
    <StudioDataView<HelixJobResponse>
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      onRowClick={(row: HelixJobResponse, _index, event) =>
        openRow(event, getJobDetailRoute(row, workspace))
      }
      attributes={{
        DataViewRoot: {
          data: pageRows,
          totalCount: recentJobs.length,
          requestStatus: isLoading || isPlaceholderData ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <EntityEmptyState
              entity="recentJobs"
              className="py-density-xl"
              variant="first-use"
              onCreate={
                CUSTOMIZER_ENABLED
                  ? () => navigate(getNewCustomizationJobRoute(workspace))
                  : undefined
              }
            />
          ),
        },
      }}
    />
  );
};
