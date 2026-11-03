// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getPartsFromReference } from '@nemo/common/src/namedEntity';
import {
  useCompassGetAnalysisRun,
  useCompassListAnalysisRuns,
} from '@nemo/sdk/generated/compass/compass-analysis-runs';
import type { AnalysisRun } from '@nemo/sdk/generated/compass/schema';
import { Text } from '@nvidia/foundations-react-core';
import { analysisJobStatus } from '@studio/api/insightsAnalysis';
import { LIST_POLL_MS, runPollInterval } from '@studio/api/useLatestAnalysisRun';
import { analysisRunBadge } from '@studio/routes/agents/AgentDetailRoute/analysis/analysisRunBadge';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { formatDateTime } from '@studio/util/date';
import { keepPreviousData } from '@tanstack/react-query';
import type { ComponentProps, FC } from 'react';

interface RunStatusProps {
  workspace: string;
  name: string;
}

/** Reads the run's backing job, polling while it is in flight. */
const RunStatus: FC<RunStatusProps> = ({ workspace, name }) => {
  const { data } = useCompassGetAnalysisRun(workspace, name, {
    query: {
      refetchInterval: ({ state }) => runPollInterval(state.data, state.fetchFailureCount),
    },
  });
  if (!data) return <Text className="text-secondary">—</Text>;
  const badge = analysisRunBadge(!!data.job, analysisJobStatus(data.job));
  return <StatusBadge status={badge.status} label={badge.label} />;
};

interface InsightJobsDataViewProps {
  workspace: string;
  agent: string;
}

/** The agent's insights analysis runs, newest first; each row opens the run's job. */
export const InsightJobsDataView: FC<InsightJobsDataViewProps> = ({ workspace, agent }) => {
  const openRow = useRowNavigation();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const sortState = dataViewState.sorting.state[0];
  const sortParam = sortState ? `${sortState.desc ? '-' : ''}${sortState.id}` : '-created_at';

  const { data, isFetching, error } = useCompassListAnalysisRuns(
    workspace,
    {
      agent,
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: sortParam,
    },
    // Picks up runs started outside this page, such as by the periodic scheduler.
    { query: { placeholderData: keepPreviousData, refetchInterval: LIST_POLL_MS } }
  );

  const makeColumns: ComponentProps<typeof StudioDataView<AnalysisRun>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('name', {
      id: 'status',
      header: 'Status',
      enableSorting: false,
      size: 120,
      cell({ row }) {
        return row.original.name ? (
          <RunStatus workspace={workspace} name={row.original.name} />
        ) : (
          <Text className="text-secondary">—</Text>
        );
      },
    }),
    accessor('name', {
      header: 'Job',
      enableSorting: true,
      size: 280,
      cell({ row }) {
        return <Text className="truncate font-bold">{row.original.name}</Text>;
      },
    }),
    accessor('since', {
      header: 'Traces since',
      enableSorting: true,
      size: 180,
      cell({ row }) {
        return (
          <Text>{row.original.since ? formatDateTime(row.original.since) : 'All history'}</Text>
        );
      },
    }),
    accessor('fast_model', {
      header: 'Fast model',
      enableSorting: true,
      size: 200,
      cell({ row }) {
        const ref = row.original.fast_model;
        return (
          <Text className="truncate">{ref ? getPartsFromReference(ref).name || ref : '—'}</Text>
        );
      },
    }),
    accessor('created_at', {
      header: 'Requested',
      enableSorting: true,
      size: 140,
      cell({ row }) {
        return row.original.created_at ? (
          <RelativeTime datetime={row.original.created_at} />
        ) : (
          <Text>—</Text>
        );
      },
    }),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      onRowClick={(row, _index, event) => {
        if (row.name) openRow(event, getWorkspaceJobDetailRoute(workspace, row.name));
      }}
      attributes={{
        DataViewRoot: {
          data: data?.data ?? [],
          totalCount: data?.pagination?.total_results,
          requestStatus: error ? 'error' : isFetching && !data ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="insightJobs"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="insightJobs" variant="first-use" />
            ),
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(error ?? new Error('Failed to fetch insights jobs'))}
            />
          ),
        },
      }}
    />
  );
};
