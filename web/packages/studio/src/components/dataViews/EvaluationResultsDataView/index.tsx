// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { type FilterTree, jsonFilter } from '@nemo/common/src/api/filterOperators';
import { CancelJobButton } from '@nemo/common/src/components/CancelJobButton';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import {
  type ApiFilter,
  useStudioDataViewState,
} from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParamWithWhitelist } from '@nemo/common/src/utils/query';
import {
  getEvaluatorListEvaluateJobsQueryKey,
  useEvaluatorListEvaluateJobs,
} from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import {
  type EvaluateJob,
  type EvaluateJobsListFilter,
  EvaluateJobsSortField,
} from '@nemo/sdk/generated/evaluator/schema';
import { Flex } from '@nvidia/foundations-react-core';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationResultDetailsRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { ComponentProps } from 'react';

const STATUS_OPTIONS_WITH_ALL = [{ value: '', label: 'All' }, ...STATUS_FILTER_OPTIONS];

const SORTABLE_FIELDS = Object.values(EvaluateJobsSortField).filter((v) => !v.startsWith('-'));
const DEFAULT_SORT = EvaluateJobsSortField['-created_at'];

const AGENT_TARGET_FORMATS = ['generic', 'nemo_agent_toolkit'];

// `$nin` alone drops runs whose spec names no target, so the null branch keeps those.
const NOT_AGENT_TRIGGERED: FilterTree<EvaluateJobsListFilter> = {
  $or: [
    { 'spec.target.format': { $nin: AGENT_TARGET_FORMATS } },
    { 'spec.target.format': { $eq: null } },
  ],
};

const buildFilter = (apiFilter: ApiFilter<EvaluateJobsListFilter>): EvaluateJobsListFilter => {
  const conditions: FilterTree<EvaluateJobsListFilter>[] = [NOT_AGENT_TRIGGERED];
  if (apiFilter.filter && Object.keys(apiFilter.filter).length > 0) {
    conditions.push(apiFilter.filter);
  }
  if (apiFilter.searchText) {
    conditions.push({ name: { $like: apiFilter.searchText } });
  }
  return jsonFilter<EvaluateJobsListFilter>(
    conditions.length === 1 ? conditions[0] : { $and: conditions }
  );
};

export const EvaluationResultsDataView = () => {
  const workspace = useWorkspaceFromPath();
  const openRow = useRowNavigation();
  const queryClient = useQueryClient();

  const dataViewState = useStudioDataViewState<EvaluateJobsListFilter>({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const {
    data: jobsData,
    isLoading,
    error,
  } = useEvaluatorListEvaluateJobs(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParamWithWhitelist(dataViewState.sorting.state, SORTABLE_FIELDS, DEFAULT_SORT),
      filter: buildFilter(dataViewState.apiFilter),
    },
    {
      query: {
        placeholderData: keepPreviousData,
        staleTime: 0,
        refetchOnWindowFocus: true,
        // Job rows change state on their own, so the list has to poll -- as
        // every sibling job list does. Without it a status badge is frozen at
        // whatever it was on page load, and Cancel appears to hang at
        // "Cancelling" even after the job has long since finished.
        refetchInterval: JOB_POLLING_INTERVAL_MS,
      },
    }
  );

  const jobs = jobsData?.data ?? [];

  const makeColumns: ComponentProps<typeof StudioDataView<EvaluateJob>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor((original) => original?.name || '', {
      id: 'name',
      header: 'Name',
    }),
    accessor((original) => original?.status || '', {
      id: 'status',
      header: 'Status',
      size: 160,
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
      size: 200,
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
            onCancelled={() =>
              queryClient.invalidateQueries({
                queryKey: getEvaluatorListEvaluateJobsQueryKey(workspace),
              })
            }
            compact
          />
        </Flex>
      ),
    }),
  ];

  const hasActiveFilters =
    !!dataViewState.debouncedSearchBar || dataViewState.debouncedColumnFilters.length > 0;
  const isInitialEmpty = jobs.length === 0 && !isLoading && !error && !hasActiveFilters;

  if (error) {
    return <ErrorPanel errorMessage={getErrorMessage(error)} />;
  }

  return (
    <StudioDataView<EvaluateJob>
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={(row, _index, event) => {
        if (!row.name) return;
        openRow(event, getEvaluationResultDetailsRoute(workspace, row.name));
      }}
      attributes={{
        DataViewSearchBar: {
          placeholder: 'Search by name',
        },
        DataViewRoot: {
          data: jobs,
          totalCount: jobsData?.pagination?.total_results ?? 0,
          // `isLoading`, not `isFetching`: this list polls, and `isFetching` is
          // true on every background refetch, which swapped the rows for
          // skeletons every few seconds. `keepPreviousData` was already holding
          // the rows -- the table was simply being told they were not there.
          requestStatus: isLoading && !jobsData ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: () =>
            isInitialEmpty ? (
              <EntityEmptyState entity="evaluationResults" variant="first-use" />
            ) : (
              <EntityEmptyState
                entity="evaluationResults"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ),
        },
      }}
    />
  );
};
