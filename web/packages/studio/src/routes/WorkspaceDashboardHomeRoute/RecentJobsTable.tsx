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

/** Running jobs, or jobs finished in the last 48 hours (`updated_at` stands in for finish time). */
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

const isRecentJob = (job: HelixJobResponse, finishedSinceMs: number): boolean => {
  if (IN_PROGRESS_JOB_STATUSES.includes(job.status)) {
    return true;
  }
  const finishedAt = utcToLocalDate(job.updated_at);
  return finishedAt !== undefined && finishedAt.getTime() >= finishedSinceMs;
};

const isJobRunning = (status: HelixJobStatus): boolean =>
  IN_PROGRESS_JOB_STATUSES.includes(status) && status !== HelixJobStatus.paused;

export const RecentJobsTable = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const openRow = useRowNavigation();
  // Fixed at mount to keep the query key stable across polls.
  const [finishedSince] = useState(() => new Date(Date.now() - RECENT_WINDOW_MS).toISOString());

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    defaultPageSize: 10,
    columnVisibility: { created_at: false },
  });

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
        if (!sourceValue) {
          return '-';
        }
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
      meta: { title: false },
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
