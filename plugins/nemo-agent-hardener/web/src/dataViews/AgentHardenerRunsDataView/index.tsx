// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  getAgentHardenerListRunsQueryKey,
  useAgentHardenerCancelJob,
  useAgentHardenerDeleteJob,
  useAgentHardenerDeleteRun,
  useAgentHardenerListRuns,
} from '@agent-hardener/generated/api';
import type { AgentHardenerRun, RunFilter } from '@agent-hardener/generated/schema';
import { useNotify, useToast, useWorkspace } from '@agent-hardener/host';
import { getAgentHardenerRunDetailsRoute } from '@agent-hardener/paths';
import { DeleteConfirmationModal, JOB_POLLING_INTERVAL_MS, QuickActionsMenuRoot, RelativeTime, StatusBadge, StudioDataView, TableEmptyState, getSortParam, useStudioDataViewState, withOperators, type StatusConfigEntry } from '@nemo/common';
import { Text } from '@nvidia/foundations-react-core';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { ComponentProps, FC, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

type AgentHardenerRunWithId = AgentHardenerRun & { id: string };

// Agent Hardener run statuses aren't platform-job statuses, so map them explicitly for the badge.
const RUN_STATUS_CONFIG: Record<string, StatusConfigEntry> = {
  running: { label: 'Running', color: 'blue' },
  completed: { label: 'Completed', color: 'green' },
  failed: { label: 'Failed', color: 'red' },
};

const STATUS_FILTER_OPTIONS = [
  { label: 'Running', value: 'running' },
  { label: 'Completed', value: 'completed' },
  { label: 'Failed', value: 'failed' },
];

export const AgentHardenerRunsDataView: FC = () => {
  const navigate = useNavigate();
  const workspace = useWorkspace();
  const toast = useToast();
  const notify = useNotify();
  const queryClient = useQueryClient();
  const [toDelete, setToDelete] = useState<AgentHardenerRunWithId | null>(null);

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const invalidateRuns = () =>
    queryClient.invalidateQueries({ queryKey: getAgentHardenerListRunsQueryKey(workspace) });

  const cancelJob = useAgentHardenerCancelJob({
    mutation: {
      onSuccess: () => {
        toast.success('War-game cancelled.');
        invalidateRuns();
      },
      onError: () => toast.error('Failed to cancel the war-game.'),
    },
  });
  const deleteRun = useAgentHardenerDeleteRun();
  const deleteJob = useAgentHardenerDeleteJob();

  const { data: runsResponse, isLoading } = useAgentHardenerListRuns(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state),
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      filter: {
        ...((dataViewState.apiFilter.filter ?? {}) as RunFilter),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<RunFilter>({ agent: { $like: dataViewState.apiFilter.searchText } })
          : {}),
      },
    },
    {
      query: {
        placeholderData: keepPreviousData,
        refetchInterval: JOB_POLLING_INTERVAL_MS,
        refetchOnMount: 'always',
        // Fail fast to the empty state instead of the app-default 3 retries (~7s of "loading")
        // when the agent-hardener service isn't reachable.
        retry: false,
      },
    }
  );

  const runs = useMemo<AgentHardenerRunWithId[]>(() => {
    const rows = (runsResponse?.data ?? []) as AgentHardenerRun[];
    return rows.map((run) => ({
      ...run,
      id: run.id || `${run.workspace ?? ''}/${run.name ?? ''}`,
    }));
  }, [runsResponse]);

  const totalResults =
    (runsResponse?.pagination as { total_results?: number } | undefined)?.total_results ?? 0;

  const makeColumns: ComponentProps<typeof StudioDataView<AgentHardenerRunWithId>>['makeColumns'] = (
    { accessor },
    { rowActionsColumn }
  ) => [
    accessor('name', { header: 'Run', cell: ({ row }) => row.original.name ?? '-' }),
    accessor('agent', {
      header: 'Agent',
      cell: ({ row }) => (
        <Text className="truncate" style={{ maxWidth: 240 }} kind="body/regular/md">
          {row.original.agent || '-'}
        </Text>
      ),
    }),
    accessor('status', {
      header: 'Status',
      size: 125,
      meta: {
        filter: { type: 'single-select' as const, label: 'Status', options: STATUS_FILTER_OPTIONS },
      },
      cell: ({ row }) => {
        if (!row.original.status) return null;
        const badge = <StatusBadge status={row.original.status} statusConfig={RUN_STATUS_CONFIG} />;
        // Surface the classified failure cause on hover for a failed run.
        return row.original.status === 'failed' && row.original.error_message ? (
          <span title={row.original.error_message}>{badge}</span>
        ) : (
          badge
        );
      },
    }),
    accessor('created_at', {
      id: 'created_at',
      header: 'Started',
      enableSorting: true,
      size: 160,
      cell: ({ row }) =>
        row.original.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
    }),
    rowActionsColumn({
      size: 70,
      cell: ({ row }) => {
        const jobId = row.original.job_id;
        return (
          <QuickActionsMenuRoot
            actions={[
              ...(row.original.status === 'running' && jobId
                ? [
                    {
                      label: 'Cancel',
                      onSelect: () => cancelJob.mutate({ workspace, name: jobId }),
                    },
                  ]
                : []),
              { label: 'Delete', onSelect: () => setToDelete(row.original) },
            ]}
          />
        );
      },
    }),
  ];

  return (
    <>
      <StudioDataView<AgentHardenerRunWithId>
        dataViewState={dataViewState}
        searchField="agent"
        makeColumns={makeColumns}
        onRowClick={(row) => row.name && navigate(getAgentHardenerRunDetailsRoute(workspace, row.name))}
        attributes={{
          DataViewSearchBar: { placeholder: 'Search by agent...' },
          DataViewRoot: {
            data: runs,
            totalCount: totalResults,
            requestStatus: isLoading && !runsResponse ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <TableEmptyState
                header="No war-game runs yet"
                emptyMessage="Agent Hardener runs appear here once you harden an agent from the CLI or a submitted job."
              />
            ),
          },
        }}
      />
      <DeleteConfirmationModal
        onNotify={notify}
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        title={`Delete ${toDelete?.name ?? 'run'}?`}
        description="This permanently deletes the run record and its platform job."
        successText="Run deleted."
        errorText="Failed to delete the run."
        onDelete={async () => {
          if (!toDelete?.name) return false;
          await deleteRun.mutateAsync({ workspace, name: toDelete.name });
          // The run's job is best-effort — the record is the user-facing artifact.
          if (toDelete.job_id)
            await deleteJob
              .mutateAsync({ workspace, name: toDelete.job_id })
              .catch(() => undefined);
          invalidateRuns();
          return true;
        }}
      />
    </>
  );
};
