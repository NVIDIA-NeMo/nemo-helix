// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FilterOperators, WithFilterOperators } from '@nemo/common/src/api/filterOperators';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useAgentsListOptimizeJobs } from '@nemo/sdk/generated/agents/agents';
import type { OptimizeJob, OptimizeJobsListFilter } from '@nemo/sdk/generated/agents/schema';
import { Banner, Text } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback } from 'react';
import { useNavigate } from 'react-router';

/** Statuses that will not change again, so polling can stop. */
const TERMINAL_STATUSES = new Set(['completed', 'error', 'cancelled']);

type OptimizeJobsFilterInput = WithFilterOperators<Omit<OptimizeJobsListFilter, 'spec'>> & {
  'spec.agent'?: FilterOperators<string>;
};

/**
 * Scope the list to one agent, optionally narrowed by a name search.
 */
const agentJobsFilter = (
  workspace: string,
  agent: string,
  search: string
): OptimizeJobsListFilter => {
  const filter: OptimizeJobsFilterInput = {
    'spec.agent': { $in: [agent, `${workspace}/${agent}`] },
  };
  if (search) filter.name = { $like: `%${search}%` };
  return JSON.stringify(filter) as unknown as OptimizeJobsListFilter;
};

interface OptimizeJobsTableProps {
  agentName?: string;
}

/**
 * Numeric HPO studies (`agents.optimize`) for one agent.
 *
 * Filtering, search and paging all run on the server: `OptimizeJobsListFilter` accepts a path into
 * the job's spec, so `spec.agent` scopes the list to this agent (see {@link agentJobsFilter}).
 *
 * One backend gap still shapes the columns: the optimize job never writes `status_details`, and
 * trial counts and best scores live in job results, which would cost one request per row. So there
 * is no Trials or Best result column, even though the design calls for both — those need the job to
 * publish a study summary first.
 */
export const OptimizeJobsTable: FC<OptimizeJobsTableProps> = ({ agentName }) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const searchText = dataViewState.debouncedSearchBar.trim();
  const { pageIndex, pageSize } = dataViewState.pagination.state;

  const { data, isPending, error } = useAgentsListOptimizeJobs(
    workspace,
    {
      page: pageIndex + 1,
      page_size: pageSize,
      sort: '-created_at',
      filter: agentJobsFilter(workspace, agentName ?? '', searchText),
    },
    {
      query: {
        enabled: !!workspace && !!agentName,
        placeholderData: keepPreviousData,
        refetchInterval: (query) => {
          const live = (query.state.data?.data ?? []).some(
            (job) => !TERMINAL_STATUSES.has(job.status ?? '')
          );
          return live ? JOB_POLLING_INTERVAL_MS : false;
        },
      },
    }
  );

  const resetFilters = useCallback(() => dataViewState.resetFilters(), [dataViewState]);

  const makeColumns: ComponentProps<typeof StudioDataView<OptimizeJob>>['makeColumns'] = (
    { accessor },
    { rowActionsColumn }
  ) => [
    accessor('name', {
      header: 'Name',
      cell: ({ row }) => <Text title={row.original.name}>{row.original.name}</Text>,
    }),
    accessor((row) => row.spec?.optimize_config, {
      id: 'optimize_config',
      header: 'Config',
      cell: ({ row }) => (
        <Text className="text-secondary max-w-[280px] truncate" kind="body/regular/sm">
          {row.original.spec?.optimize_config ?? '—'}
        </Text>
      ),
    }),
    accessor('status', {
      header: 'Status',
      size: 125,
      cell: ({ row }) =>
        row.original.status ? <StatusBadge status={row.original.status} /> : null,
    }),
    accessor('created_at', {
      id: 'created_at',
      header: 'Created',
      size: 150,
      cell: ({ row }) =>
        row.original.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
    }),
    rowActionsColumn({ size: 70, enableResizing: false }),
  ];

  return (
    <>
      {error && (
        <Banner kind="inline" status="error">
          {error instanceof Error ? error.message : 'Could not load optimization studies.'}
        </Banner>
      )}

      <StudioDataView<OptimizeJob>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row) => navigate(getWorkspaceJobDetailRoute(workspace, row.name))}
        attributes={{
          DataViewSearchBar: { placeholder: 'Search by name...' },
          DataViewRoot: {
            data: data?.data ?? [],
            totalCount: data?.pagination?.total_results ?? 0,
            requestStatus: isPending && !!agentName ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () =>
              searchText ? (
                <EntityEmptyState
                  entity="agentOptimizations"
                  variant="no-results"
                  onClearFilters={resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="agentOptimizations"
                  variant="first-use"
                  workspace={workspace}
                />
              ),
          },
        }}
      />
    </>
  );
};
