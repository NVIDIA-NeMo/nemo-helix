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
  getDataDesignerListCreateJobsQueryKey,
  useDataDesignerDeleteCreateJob,
  useDataDesignerListCreateJobs,
} from '@nemo/sdk/generated/data-designer/data-designer';
import type {
  CreateJob as DataDesignerJob,
  CreateJobsListFilter as DataDesignerJobsListFilter,
  CreateJobsSortField as DataDesignerJobsSortField,
} from '@nemo/sdk/generated/data-designer/schema';
import { Banner, Button, Text } from '@nvidia/foundations-react-core';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { DataDesignerJobActionsMenu } from '@studio/components/DataDesignerJobActionsMenu';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getDataDesignerJobDetailsRoute, getNewDataDesignerJobRoute } from '@studio/routes/utils';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { ComponentProps, FC, useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router';

type DataDesignerJobWithId = DataDesignerJob & { id: string };

export const DataDesignerJobsDataView: FC = () => {
  const navigate = useNavigate();
  const openRow = useRowNavigation();
  const workspace = useWorkspaceFromPath();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    columnVisibility: { updated_at: false },
  });

  const queryClient = useQueryClient();

  const [deleteJobs, setDeleteJobs] = useState<DataDesignerJob[]>([]);
  const [cancelError, setCancelError] = useState<string | undefined>(undefined);

  const deleteJobMutation = useDataDesignerDeleteCreateJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getDataDesignerListCreateJobsQueryKey(workspace),
        }),
    },
  });

  const handleDeleteJobs = async (jobsToDelete: DataDesignerJob[]) => {
    const invalid = jobsToDelete.filter((job) => !job.workspace || !job.name);
    if (invalid.length > 0) {
      throw new Error(
        `Cannot delete ${invalid.length} job${invalid.length !== 1 ? 's' : ''}: missing workspace or name.`
      );
    }
    await Promise.all(
      jobsToDelete.map(async (job) => {
        try {
          await deleteJobMutation.mutateAsync({ workspace: job.workspace!, name: job.name });
        } catch (error) {
          throw new Error(
            `Failed to delete job "${job.name}": ${error instanceof Error ? error.message : 'Unknown error'}`
          );
        }
      })
    );
  };

  const { data: dataDesignerResponse, isLoading } = useDataDesignerListCreateJobs(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state) as DataDesignerJobsSortField,
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      filter: {
        ...((dataViewState.apiFilter.filter ?? {}) as DataDesignerJobsListFilter),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<DataDesignerJobsListFilter>({
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

  const jobs = useMemo<DataDesignerJobWithId[]>(
    () =>
      (dataDesignerResponse?.data || []).map((job) => ({
        ...job,
        id: job.id || `${job.workspace ?? ''}/${job.name}`,
      })),
    [dataDesignerResponse?.data]
  );

  const hasActiveFilters =
    Boolean(dataViewState.debouncedSearchBar) || dataViewState.debouncedColumnFilters.length > 0;

  const resetFilters = useCallback(() => {
    dataViewState.resetFilters();
  }, [dataViewState]);

  const makeColumns: ComponentProps<typeof StudioDataView<DataDesignerJobWithId>>['makeColumns'] = (
    { accessor },
    { rowSelectionColumn, rowActionsColumn }
  ) => [
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
        <DataDesignerJobActionsMenu
          job={row.original}
          includeViewDetails
          onCancelError={setCancelError}
        />
      ),
    }),
  ];

  const totalResults = dataDesignerResponse?.pagination?.total_results ?? 0;

  return (
    <>
      {cancelError && (
        <Banner kind="inline" status="error">
          {cancelError}
        </Banner>
      )}

      <StudioDataView<DataDesignerJobWithId>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row, _index, event) =>
          openRow(event, getDataDesignerJobDetailsRoute(workspace, row.name))
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
            requestStatus: isLoading && !dataDesignerResponse ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () =>
              hasActiveFilters ? (
                <EntityEmptyState
                  entity="dataDesignerJobs"
                  variant="no-results"
                  onClearFilters={resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="dataDesignerJobs"
                  variant="first-use"
                  onCreate={() => navigate(getNewDataDesignerJobRoute(workspace))}
                />
              ),
          },
        }}
      />

      <BulkDeleteModal
        items={deleteJobs}
        open={deleteJobs.length > 0}
        onDelete={handleDeleteJobs}
        title={(count) => `Delete ${count} Data Designer Job${count !== 1 ? 's' : ''}`}
        onClose={() => {
          setDeleteJobs([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </>
  );
};
