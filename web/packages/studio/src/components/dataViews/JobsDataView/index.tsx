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
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  PlatformJobListSortField,
  PlatformJobResponse,
  PlatformJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { Flex, Tag } from '@nvidia/foundations-react-core';
import {
  HIDDEN_JOB_SOURCES,
  JOB_SOURCE,
  SOURCE_OPTIONS,
} from '@studio/components/dataViews/JobsDataView/constants';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { iconColorClass } from '@studio/routes/constants';
import { keepPreviousData } from '@tanstack/react-query';
import { ChartBar, Cog, LayoutList, Sliders, Sparkles } from 'lucide-react';
import { ComponentProps, type ReactNode, useRef } from 'react';

const SOURCE_DISPLAY: Record<string, { label: string; icon: ReactNode }> = {
  [JOB_SOURCE.CUSTOMIZATION]: {
    label: 'Customizer',
    icon: <Sliders className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.DATA_DESIGNER]: {
    label: 'Data Designer',
    icon: <LayoutList className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.SAFE_SYNTHESIZER]: {
    label: 'Safe Synthesizer',
    icon: <Sparkles className={iconColorClass} size={14} />,
  },
  [JOB_SOURCE.EVALUATOR_METRICS]: {
    label: 'Evaluator',
    icon: <ChartBar className={iconColorClass} size={14} />,
  },
};

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

  const hiddenJobSources = CUSTOMIZER_ENABLED
    ? HIDDEN_JOB_SOURCES
    : [...HIDDEN_JOB_SOURCES, JOB_SOURCE.CUSTOMIZATION];
  const sourceFilterOptions = SOURCE_OPTIONS.filter(
    (option) => CUSTOMIZER_ENABLED || option.value !== JOB_SOURCE.CUSTOMIZATION
  );
  const hasUserSourceFilter = userFilter.source !== undefined && userFilter.source !== '';

  const {
    data: jobsData,
    isFetching,
    error,
  } = useJobsListJobs(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParam(dataViewState.sorting.state) as PlatformJobListSortField,
      filter: {
        ...userFilter,
        ...(hasUserSourceFilter
          ? {}
          : withOperators<PlatformJobsListFilter>({ source: { $nin: hiddenJobSources } })),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<PlatformJobsListFilter>({
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
    .map((source) => ({ label: SOURCE_DISPLAY[source]?.label ?? source, value: source }));
  const mergedSourceOptions = [...sourceFilterOptions, ...dynamicSourceOptions];

  const makeColumns: ComponentProps<typeof StudioDataView<PlatformJobResponse>>['makeColumns'] = ({
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
        filter: {
          type: 'single-select' as const,
          label: 'Source',
          options: mergedSourceOptions,
        },
      },
      cell: ({ row, column: col }) => {
        const sourceValue = row.original.source;
        if (!sourceValue) return '-';
        const display = SOURCE_DISPLAY[sourceValue];
        const icon = display?.icon ?? <Cog className={iconColorClass} size={14} />;
        const label = display?.label ?? sourceValue;
        return (
          <Tag
            kind="outline"
            color="gray"
            density="compact"
            data-no-row-click
            onClick={() => col.setFilterValue(sourceValue)}
          >
            <Flex align="center" gap="density-sm">
              {icon}
              {label}
            </Flex>
          </Tag>
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
    return <ErrorPanel errorMessage={getErrorMessage(error)} />;
  }

  return (
    <StudioDataView<PlatformJobResponse>
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={(row: PlatformJobResponse, _index, event) =>
        openRow(event, getJobDetailRoute(row, workspace))
      }
      attributes={{
        DataViewSearchBar: {
          placeholder: 'Search by name',
        },
        DataViewRoot: {
          data: jobs,
          totalCount: CUSTOMIZER_ENABLED ? jobsData?.pagination?.total_results || 0 : jobs.length,
          requestStatus: isFetching ? 'loading' : undefined,
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
