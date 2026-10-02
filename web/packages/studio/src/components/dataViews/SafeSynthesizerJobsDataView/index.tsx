// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { QuickActionsMenuRoot } from '@nemo/common/src/components/QuickActionsMenu/QuickActionsMenuRoot';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { ScoreGauge } from '@nemo/common/src/components/ScoreGauge';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useJobsCancelJob, useJobsDeleteJob } from '@nemo/sdk/generated/platform/jobs';
import {
  getSafeSynthesizerDownloadJobResultSummaryQueryOptions as getDownloadJobResultSummaryQueryOptions,
  getSafeSynthesizerListJobsQueryKey,
  useSafeSynthesizerListJobs,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import {
  GenerateJob,
  GenerateJobsListFilter,
  GenerateJobsSortField,
} from '@nemo/sdk/generated/safe-synthesizer/schema';
import { Banner, Button, Stack } from '@nvidia/foundations-react-core';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import { isCancellableJob } from '@studio/components/dataViews/SafeSynthesizerJobsDataView/utils';
import { FilesetFilePreviewLink } from '@studio/components/SafeSynthesizerFilesetPreview/FilesetFilePreviewLink';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import {
  getNewSafeSynthesizerRoute,
  getGenerateJobReportRoute,
  getGenerateJobRoute,
} from '@studio/routes/utils';
import { keepPreviousData, useQueries, useQueryClient } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { ComponentProps, FC, useCallback, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router';

type GenerateJobWithId = GenerateJob & { id: string };

export const GenerateJobsDataView: FC = () => {
  const navigate = useNavigate();
  const openRow = useRowNavigation();
  const workspace = useWorkspaceFromPath();
  const queryClient = useQueryClient();

  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const [deleteJobs, setDeleteJobs] = useState<GenerateJob[]>([]);
  const [cancelError, setCancelError] = useState<string | undefined>(undefined);

  const deleteJobMutation = useJobsDeleteJob({
    mutation: {
      onSuccess: () =>
        queryClient.resetQueries({
          queryKey: getSafeSynthesizerListJobsQueryKey(workspace),
        }),
    },
  });

  const handleDeleteJobs = async (jobsToDelete: GenerateJob[]) => {
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

  // Cancel job mutation
  const cancelJobMutation = useJobsCancelJob({
    mutation: {
      onSuccess: () => {
        queryClient.resetQueries({
          queryKey: getSafeSynthesizerListJobsQueryKey(workspace),
        });
        setCancelError(undefined);
      },
      onError: (error) => {
        setCancelError(error instanceof Error ? error.message : 'Failed to cancel job');
      },
    },
  });

  const handleCancelJob = useCallback(
    async (job: GenerateJob) => {
      if (job.workspace && job.name) {
        try {
          setCancelError(undefined);
          await cancelJobMutation.mutateAsync({ workspace: job.workspace, name: job.name });
        } catch {
          // Error is handled by onError callback
        }
      }
    },
    [cancelJobMutation]
  );

  // Fetch jobs using dataViewState for pagination, sorting, search, and filters
  const { data: safeSynthesizerResponse, isLoading } = useSafeSynthesizerListJobs(
    workspace,
    {
      sort: getSortParam(dataViewState.sorting.state) as GenerateJobsSortField,
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      filter: {
        ...((dataViewState.apiFilter.filter ?? {}) as GenerateJobsListFilter),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<GenerateJobsListFilter>({
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

  // Filter jobs with valid IDs for summary queries
  const jobsWithIds = useMemo(
    () =>
      (safeSynthesizerResponse?.data || []).filter(
        (row): row is GenerateJobWithId => row.id !== undefined
      ),
    [safeSynthesizerResponse?.data]
  );

  // Fetch summary data for each row that has completed status
  const summaryQueries = useQueries({
    queries:
      safeSynthesizerResponse?.data
        .filter((row) => row.name !== undefined && row.workspace !== undefined)
        .map((row) =>
          getDownloadJobResultSummaryQueryOptions(row.workspace!, row.name!, {
            query: {
              enabled: row.status === 'completed',
              staleTime: 10 * 60 * 1000, // 10 minutes
              gcTime: 10 * 60 * 1000, // 10 minutes
            },
          })
        ) ?? [],
  });

  // Ensure each job has a unique id for DataView row selection
  const jobs = useMemo<GenerateJobWithId[]>(
    () =>
      (safeSynthesizerResponse?.data || []).map((job) => ({
        ...job,
        id: job.id || `${job.workspace}/${job.name}`,
      })),
    [safeSynthesizerResponse?.data]
  );

  // Create a map of job id to summary query index for efficient lookup
  const summaryDataMap = useMemo(() => {
    const map = new Map<string, number>();
    jobsWithIds.forEach((row, index) => {
      map.set(row.id, index);
    });
    return map;
  }, [jobsWithIds]);

  const hasActiveFilters =
    !!dataViewState.debouncedSearchBar || dataViewState.debouncedColumnFilters.length > 0;

  // Column definitions
  const makeColumns: ComponentProps<typeof StudioDataView<GenerateJobWithId>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({
          size: ROW_SELECTION_COLUMN_SIZE,
        }),
        accessor('name', {
          header: 'Name',
        }),
        {
          id: 'fileset',
          header: 'Dataset',
          cell: ({ row }: { row: DataView.TanstackTable.Row<GenerateJobWithId> }) => (
            <FilesetFilePreviewLink url={row.original.spec?.data_source as string}>
              <span className="truncate font-semibold text-sm">
                {row.original.spec?.data_source as string}
              </span>
            </FilesetFilePreviewLink>
          ),
        },
        {
          id: 'sqs',
          header: () => (
            <abbr title="Synthetic Quality Score" className="no-underline">
              SQS
            </abbr>
          ),
          size: 70,
          cell: ({ row }: { row: DataView.TanstackTable.Row<GenerateJobWithId> }) => {
            const summaryIndex = summaryDataMap.get(row.original.id);
            const summaryData =
              summaryIndex !== undefined ? summaryQueries[summaryIndex]?.data : undefined;
            return (
              <Link
                to={getGenerateJobReportRoute(workspace, row.original.name!)}
                className="flex items-center"
                aria-label={`View SQS for job ${row.original.name}`}
              >
                <ScoreGauge score={summaryData?.synthetic_data_quality_score} size="sm" />
              </Link>
            );
          },
        },
        {
          id: 'dps',
          header: () => (
            <abbr title="Data Privacy Score" className="no-underline">
              DPS
            </abbr>
          ),
          size: 70,
          cell: ({ row }: { row: DataView.TanstackTable.Row<GenerateJobWithId> }) => {
            const summaryIndex = summaryDataMap.get(row.original.id);
            const summaryData =
              summaryIndex !== undefined ? summaryQueries[summaryIndex]?.data : undefined;
            return (
              <Link
                to={getGenerateJobReportRoute(workspace, row.original.name!)}
                className="flex items-center"
                aria-label={`View DPS for job ${row.original.name}`}
              >
                <ScoreGauge score={summaryData?.data_privacy_score} size="sm" />
              </Link>
            );
          },
        },
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
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          cell: ({ row }) => (
            <QuickActionsMenuRoot
              actions={[
                {
                  label: 'View Summary',
                  onSelect: () => {
                    if (row.original.name) {
                      navigate(getGenerateJobRoute(workspace, row.original.name));
                    }
                  },
                },
                ...(row.original.status === 'completed'
                  ? [
                      {
                        label: 'View Report',
                        onSelect: () => {
                          if (row.original.name) {
                            navigate(getGenerateJobReportRoute(workspace, row.original.name));
                          }
                        },
                      },
                    ]
                  : []),
                {
                  label: 'Delete',
                  onSelect: () => setDeleteJobs([row.original]),
                },
                ...(isCancellableJob(row.original.status)
                  ? [
                      {
                        label: 'Cancel',
                        onSelect: () => handleCancelJob(row.original),
                      },
                    ]
                  : []),
              ]}
            />
          ),
        }),
      ],
      [handleCancelJob, navigate, summaryDataMap, summaryQueries, workspace]
    );

  const totalResults = safeSynthesizerResponse?.pagination?.total_results ?? 0;

  return (
    <Stack className="flex-1 min-h-0">
      {cancelError && (
        <Banner kind="inline" status="error">
          {cancelError}
        </Banner>
      )}

      <StudioDataView<GenerateJobWithId>
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={(row: GenerateJobWithId, _index, event) => {
          if (row.name) {
            openRow(event, getGenerateJobRoute(workspace, row.name));
          }
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
            placeholder: 'Search Jobs...',
          },
          DataViewRoot: {
            data: jobs,
            totalCount: totalResults,
            requestStatus: isLoading && !safeSynthesizerResponse ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: () =>
              hasActiveFilters ? (
                <EntityEmptyState
                  entity="safeSynthesizerJobs"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState
                  entity="safeSynthesizerJobs"
                  variant="first-use"
                  onCreate={() => navigate(getNewSafeSynthesizerRoute(workspace))}
                />
              ),
          },
        }}
      />

      <BulkDeleteModal
        items={deleteJobs}
        open={deleteJobs.length > 0}
        onDelete={handleDeleteJobs}
        title={(count) => `Delete ${count} Safe Synthesizer Job${count !== 1 ? 's' : ''}`}
        onClose={() => {
          setDeleteJobs([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </Stack>
  );
};
