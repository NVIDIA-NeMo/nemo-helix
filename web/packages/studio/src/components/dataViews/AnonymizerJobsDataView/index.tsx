// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import {
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import {
  getAnonymizerListRunJobsQueryKey,
  useAnonymizerDeleteRunJob,
  useAnonymizerListRunJobs,
} from '@nemo/sdk/generated/anonymizer/anonymizer';
import type {
  RunJob as AnonymizerJob,
  RunJobsListFilter as AnonymizerJobsListFilter,
  RunJobsSortField as AnonymizerJobsSortField,
} from '@nemo/sdk/generated/anonymizer/schema';
import { Banner, Button, Text } from '@nvidia/foundations-react-core';
import { AnonymizerJobActionsMenu } from '@studio/components/AnonymizerJobActionsMenu';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getAnonymizerJobRoute, getNewAnonymizerRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

type AnonymizerJobWithId = AnonymizerJob & { id: string };

export const AnonymizerJobsDataView: FC = () => {
  const navigate = useNavigate();
  const openRow = useRowNavigation();
  const workspace = useWorkspaceFromPath();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    columnVisibility: { updated_at: false },
  });

  const queryClient = useQueryClient();

  const [deleteJobs, setDeleteJobs] = useState<AnonymizerJobWithId[]>([]);
  const [cancelError, setCancelError] = useState<string | undefined>(undefined);

  const deleteJobMutation = useAnonymizerDeleteRunJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getAnonymizerListRunJobsQueryKey(workspace),
        }),
    },
  });

  const handleDeleteJobs = async (jobsToDelete: AnonymizerJobWithId[]) => {
    const invalid = jobsToDelete.filter((job) => !job.workspace || !job.name);
    if (invalid.length > 0) {
      throw new Error(
        `Cannot delete ${invalid.length} job${invalid.length !== 1 ? 's' : ''}: missing workspace or name.`
      );
    }
    const results = await Promise.allSettled(
      jobsToDelete.map((job) =>
        deleteJobMutation.mutateAsync({ workspace: job.workspace!, name: job.name })
      )
    );
    const failed = jobsToDelete.filter((_, i) => results[i].status === 'rejected');
    if (failed.length > 0) {
      // Keep only the failed jobs selected so a retry doesn't re-delete succeeded ones.
      dataViewState.rowSelection.set(Object.fromEntries(failed.map((job) => [job.id, true])));
      throw new Error(
        `Failed to delete ${failed.length} of ${jobsToDelete.length} job${
          jobsToDelete.length !== 1 ? 's' : ''
        }: ${failed.map((job) => `"${job.name}"`).join(', ')}`
      );
    }
  };

  const { data: anonymizerResponse, isLoading } = useAnonymizerListRunJobs(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state) as AnonymizerJobsSortField,
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      filter: {
        ...((dataViewState.apiFilter.filter ?? {}) as AnonymizerJobsListFilter),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<AnonymizerJobsListFilter>({
              name: { $like: dataViewState.apiFilter.searchText },
            })
          : {}),
      },
    },
    {
      query: {
        placeholderData: keepPreviousData,
        refetchInterval: JOB_POLLING_INTERVAL_MS,
        refetchOnMount: 'always',
      },
    }
  );

  const jobs = useMemo<AnonymizerJobWithId[]>(
    () =>
      (anonymizerResponse?.data || []).map((job) => ({
        ...job,
        id: job.id || `${job.workspace ?? ''}/${job.name}`,
      })),
    [anonymizerResponse?.data]
  );

  const hasActiveFilters =
    Boolean(dataViewState.debouncedSearchBar) || dataViewState.debouncedColumnFilters.length > 0;

  const makeColumns: ComponentProps<typeof StudioDataView<AnonymizerJobWithId>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
        accessor('name', {
          header: 'Name',
          cell: ({ row }) => row.original.name,
        }),
        accessor('description', {
          header: 'Description',
          cell: ({ row }) => (
            <Text className="max-w-[200px] truncate" kind="body/regular/md">
              {row.original.description ?? '-'}
            </Text>
          ),
        }),
        accessor('created_at', {
          id: 'created_at',
          header: 'Created',
          enableSorting: true,
          size: 150,
          meta: {
            filter: dateTimeFilter('Created At'),
          },
          cell: ({ row }) =>
            row.original.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
        }),
        accessor('status', {
          header: 'Status',
          size: 125,
          meta: {
            filter: {
              type: 'single-select' as const,
              label: 'Status',
              options: STATUS_FILTER_OPTIONS,
            },
          },
          cell: ({ row }) =>
            row.original.status ? <StatusBadge status={row.original.status} /> : null,
        }),
        accessor('updated_at', {
          id: 'updated_at',
          header: 'Updated',
          enableSorting: false,
          meta: {
            filter: dateTimeFilter('Updated At'),
          },
          cell: ({ row }) =>
            row.original?.updated_at ? <RelativeTime datetime={row.original.updated_at} /> : null,
        }),
        rowActionsColumn({
          size: 70,
          enableResizing: false,
          cell: ({ row }) => (
            <AnonymizerJobActionsMenu
              job={row.original}
              includeViewDetails
              onCancelError={setCancelError}
            />
          ),
        }),
      ],
      []
    );

  const totalResults = anonymizerResponse?.pagination?.total_results ?? 0;

  return (
    <>
      {cancelError && (
        <Banner kind="inline" status="error">
          {cancelError}
        </Banner>
      )}

      <StudioDataView<AnonymizerJobWithId>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row, _index, event) =>
          openRow(event, getAnonymizerJobRoute(workspace, row.name))
        }
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
            placeholder: 'Search jobs...',
          },
          DataViewRoot: {
            data: jobs,
            totalCount: totalResults,
            requestStatus: isLoading && !anonymizerResponse ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () =>
              hasActiveFilters ? (
                <EntityEmptyState
                  entity="anonymizerJobs"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="anonymizerJobs"
                  variant="first-use"
                  onCreate={() => navigate(getNewAnonymizerRoute(workspace))}
                />
              ),
          },
        }}
      />

      <BulkDeleteModal
        items={deleteJobs}
        open={deleteJobs.length > 0}
        onDelete={handleDeleteJobs}
        title={(count) => `Delete ${count} Anonymizer Job${count !== 1 ? 's' : ''}`}
        onClose={() => {
          setDeleteJobs([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </>
  );
};
