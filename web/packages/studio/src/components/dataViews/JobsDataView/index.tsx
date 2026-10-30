// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { CancelJobButton } from '@nemo/common/src/components/CancelJobButton';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam, getRequestStatus } from '@nemo/common/src/utils/query';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobListSortField,
  HelixJobResponse,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { Flex } from '@nvidia/foundations-react-core';
import {
  getHiddenJobSources,
  JOB_SOURCE,
  SOURCE_OPTIONS,
} from '@studio/components/dataViews/JobsDataView/constants';
import {
  getSourceDisplay,
  getSourceTitle,
} from '@studio/components/dataViews/JobsDataView/sourceDisplay';
import { SourceTag } from '@studio/components/dataViews/JobsDataView/SourceTag';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, useRef } from 'react';

const STATUS_OPTIONS_WITH_ALL = [{ value: '', label: 'All' }, ...STATUS_FILTER_OPTIONS];

export const JobsDataView = () => {
  const workspace = useWorkspaceFromPath();
  const openRow = useRowNavigation();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    defaultPageSize: 10,
  });

  const userFilter = { ...(dataViewState.apiFilter.filter ?? {}) };
  if (!CUSTOMIZER_ENABLED && userFilter.source === JOB_SOURCE.CUSTOMIZATION) {
    delete userFilter.source;
  }

  const hiddenJobSources = getHiddenJobSources();
  const sourceFilterOptions = SOURCE_OPTIONS.filter(
    (option) => CUSTOMIZER_ENABLED || option.value !== JOB_SOURCE.CUSTOMIZATION
  );
  const hasUserSourceFilter = userFilter.source !== undefined && userFilter.source !== '';

  const {
    data: jobsData,
    isLoading,
    isPlaceholderData,
    error,
    refetch,
  } = useJobsListJobs(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParam(dataViewState.sorting.state) as HelixJobListSortField,
      filter: {
        ...userFilter,
        ...(hasUserSourceFilter
          ? {}
          : withOperators<HelixJobsListFilter>({ source: { $nin: hiddenJobSources } })),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<HelixJobsListFilter>({
              name: { $like: dataViewState.apiFilter.searchText },
            })
          : {}),
      },
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

  const jobs =
    CUSTOMIZER_ENABLED || !jobsData?.data
      ? (jobsData?.data ?? [])
      : jobsData.data.filter((job) => job.source !== JOB_SOURCE.CUSTOMIZATION);

  // Surface plugin sources not in the static list (e.g. nemo-agents-plugin).
  const seenSourcesRef = useRef<Set<string>>(new Set());
  for (const job of jobsData?.data ?? []) {
    if (job.source) seenSourcesRef.current.add(job.source);
  }

  const staticSourceValues = new Set(sourceFilterOptions.map((option) => option.value));
  const dynamicSourceOptions = [...seenSourcesRef.current]
    .filter((source) => !staticSourceValues.has(source) && !hiddenJobSources.includes(source))
    .sort()
    .map((source) => ({ label: getSourceDisplay(source).label, value: source }));
  const mergedSourceOptions = [...sourceFilterOptions, ...dynamicSourceOptions];

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
      enableSorting: true,
      meta: {
        // The cell renders a label for the source; without this the OS tooltip falls back
        // to the raw accessor value and shows the API's own word for it instead.
        title: (cell) => getSourceTitle(cell.getValue()),
        filter: {
          type: 'single-select' as const,
          label: 'Source',
          options: mergedSourceOptions,
        },
      },
      cell: ({ row, column: col }) => {
        const sourceValue = row.original.source;
        if (!sourceValue) return '-';
        return (
          <SourceTag
            source={sourceValue}
            data-no-row-click
            onClick={() => col.setFilterValue(sourceValue)}
          />
        );
      },
    }),
    accessor((original) => original?.status || '', {
      id: 'status',
      header: 'Status',
      meta: {
        filter: {
          type: 'single-select' as const,
          label: 'Status',
          options: STATUS_OPTIONS_WITH_ALL,
        },
      },
      cell: ({ row }) => <StatusBadge status={row.original.status} />,
    }),
    accessor((original) => original?.created_at || '', {
      id: 'created_at',
      header: 'Created',
      enableSorting: true,
      meta: {
        filter: dateTimeFilter('Created At'),
      },
      cell: ({ row }) => <RelativeTime datetime={row.original.created_at ?? ''} />,
    }),
    accessor(() => '', {
      id: 'actions',
      header: '',
      enableSorting: false,
      cell: ({ row }) => (
        <Flex justify="end">
          <CancelJobButton
            workspace={workspace}
            jobName={row.original.name}
            jobStatus={row.original.status}
            compact
          />
        </Flex>
      ),
    }),
  ];

  if (error) {
    return (
      <ErrorPanel
        error={error}
        errorMessage={getErrorMessage(error)}
        onRetry={() => void refetch()}
      />
    );
  }

  return (
    <StudioDataView<HelixJobResponse>
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={(row: HelixJobResponse, _index, event) =>
        openRow(event, getJobDetailRoute(row, workspace))
      }
      attributes={{
        DataViewSearchBar: {
          placeholder: 'Search by name',
        },
        DataViewRoot: {
          data: jobs,
          totalCount: CUSTOMIZER_ENABLED ? jobsData?.pagination?.total_results || 0 : jobs.length,
          requestStatus: getRequestStatus({ isLoading, isPlaceholderData }),
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="jobs"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="jobs" variant="first-use" />
            ),
        },
      }}
    />
  );
};
