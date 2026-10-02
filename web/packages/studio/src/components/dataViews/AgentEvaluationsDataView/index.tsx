// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import {
  EntityEmptyState,
  type EntityEmptyStateProps,
} from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParamWithWhitelist } from '@nemo/common/src/utils/query';
import {
  getEvaluatorListAgentEvaluateJobsQueryKey,
  useEvaluatorDeleteAgentEvaluateJob,
  useEvaluatorListAgentEvaluateJobs,
} from '@nemo/sdk/generated/evaluator/evaluator-plugin-agent-eval-jobs-routes';
import {
  type AgentEvaluateJob,
  type AgentEvaluateJobsListFilter,
  AgentEvaluateJobsSortField,
} from '@nemo/sdk/generated/evaluator/schema';
import { Button } from '@nvidia/foundations-react-core';
import { agentNameForJob, evalConfigName } from '@studio/api/evaluation/agent-evaluations';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getAgentEvaluationDetailRoute, getFilesetRoute } from '@studio/routes/utils';
import { getTextWithCount } from '@studio/util/strings';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { ComponentProps, useMemo, useState } from 'react';
import { Link } from 'react-router';

type AgentEvalJobRow = AgentEvaluateJob & { id: string };

/** Variant-specific `EntityEmptyState` props for the `agentEvaluations` entity, minus `entity`/`className`. */
type AgentEvaluationsEmptyStateProps =
  | Omit<Extract<EntityEmptyStateProps, { variant: 'first-use' }>, 'entity' | 'className'>
  | Omit<Extract<EntityEmptyStateProps, { variant: 'no-results' }>, 'entity' | 'className'>;

const STATUS_OPTIONS_WITH_ALL = [{ value: '', label: 'All' }, ...STATUS_FILTER_OPTIONS];

const SORTABLE_FIELDS = Object.values(AgentEvaluateJobsSortField).filter((v) => !v.startsWith('-'));
const DEFAULT_SORT = AgentEvaluateJobsSortField['-created_at'];

export const AgentEvaluationsDataView = () => {
  const workspace = useWorkspaceFromPath();
  const openRow = useRowNavigation();
  const queryClient = useQueryClient();

  const [deleteJobs, setDeleteJobs] = useState<AgentEvalJobRow[]>([]);

  const dataViewState = useStudioDataViewState<AgentEvaluateJobsListFilter>({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const deleteJobMutation = useEvaluatorDeleteAgentEvaluateJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getEvaluatorListAgentEvaluateJobsQueryKey(workspace),
        }),
    },
  });

  const handleDeleteJobs = async (jobsToDelete: AgentEvalJobRow[]) => {
    const results = await Promise.allSettled(
      jobsToDelete.map((job) => deleteJobMutation.mutateAsync({ workspace, name: job.name }))
    );
    const failures = results.flatMap((result, i) =>
      result.status === 'rejected'
        ? [
            `"${jobsToDelete[i].name}": ${result.reason instanceof Error ? result.reason.message : 'Unknown error'}`,
          ]
        : []
    );
    if (failures.length > 0) {
      throw new Error(
        `Failed to delete ${failures.length} evaluation${failures.length !== 1 ? 's' : ''}: ${failures.join('; ')}`
      );
    }
  };

  const {
    data: jobsData,
    isLoading,
    error,
  } = useEvaluatorListAgentEvaluateJobs(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParamWithWhitelist(
        dataViewState.sorting.state,
        SORTABLE_FIELDS,
        DEFAULT_SORT
      ) as AgentEvaluateJobsSortField,
      filter: {
        ...dataViewState.apiFilter.filter,
        ...(dataViewState.apiFilter.searchText
          ? withOperators<AgentEvaluateJobsListFilter>({
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
        refetchInterval: JOB_POLLING_INTERVAL_MS,
      },
    }
  );

  const jobs = useMemo<AgentEvalJobRow[]>(
    () => (jobsData?.data ?? []).map((job) => ({ ...job, id: job.id || job.name })),
    [jobsData?.data]
  );

  const makeColumns: ComponentProps<typeof StudioDataView<AgentEvalJobRow>>['makeColumns'] = (
    { accessor },
    { rowSelectionColumn, rowActionsColumn }
  ) => [
    rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
    accessor((original) => (original ? (evalConfigName(original) ?? '') : ''), {
      id: 'eval_config',
      header: 'Eval Config',
      size: 200,
      enableSorting: false,
      cell: ({ row }) => {
        const configName = evalConfigName(row.original);
        return configName ? (
          <Link
            to={getFilesetRoute(workspace, configName)}
            className="text-primary underline"
            onClick={(e) => e.stopPropagation()}
          >
            {configName}
          </Link>
        ) : null;
      },
    }),
    accessor((original) => (original ? (agentNameForJob(original) ?? '') : ''), {
      id: 'agent',
      header: 'Agent',
      size: 200,
      enableSorting: false,
    }),
    accessor((original) => original?.name || '', {
      id: 'name',
      header: 'Job Name',
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
      cell: ({ row }) => <RelativeTime datetime={row.original.created_at ?? ''} />,
    }),
    rowActionsColumn({
      size: ROW_ACTIONS_COLUMN_SIZE,
      enableResizing: false,
      cell: ({ row }) => (
        <QuickActionsMenuRoot
          actions={[{ label: 'Delete', onSelect: () => setDeleteJobs([row.original]) }]}
        />
      ),
    }),
  ];

  if (error) {
    return (
      <ErrorPanel
        errorMessage={getErrorMessage(error ?? new Error('Failed to fetch evaluations'))}
      />
    );
  }

  return (
    <>
      <StudioDataView<AgentEvalJobRow>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row, _index, event) => {
          if (!row.name) return;
          openRow(event, getAgentEvaluationDetailRoute(workspace, row.name));
        }}
        renderBulkActions={({ selectedRows }) => (
          <Button
            kind="tertiary"
            aria-label="Delete selected jobs"
            onClick={() => setDeleteJobs(selectedRows)}
          >
            <Trash /> Delete
          </Button>
        )}
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search by name',
          },
          DataViewRoot: {
            data: jobs,
            totalCount: jobsData?.pagination?.total_results ?? 0,
            requestStatus: isLoading && !jobsData ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) => {
              const emptyStateProps: AgentEvaluationsEmptyStateProps =
                hasFiltersApplied || hasSearchApplied
                  ? { variant: 'no-results', onClearFilters: dataViewState.resetFilters }
                  : { variant: 'first-use' };
              return <EntityEmptyState entity="agentEvaluations" {...emptyStateProps} />;
            },
          },
        }}
      />

      <BulkDeleteModal
        items={deleteJobs}
        open={deleteJobs.length > 0}
        onDelete={handleDeleteJobs}
        title={(count) => `Delete ${getTextWithCount('Evaluation', count)}`}
        onClose={() => {
          setDeleteJobs([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </>
  );
};
