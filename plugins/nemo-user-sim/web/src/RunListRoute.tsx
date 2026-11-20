// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  DeleteConfirmationModal,
  RelativeTime,
  StatusBadge,
  StudioDataView,
  TableEmptyState,
  useStudioDataViewState,
} from '@nemo/common';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { useEffect, useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router';
import { isFinished, useCancelRuns, useDeleteRuns, useSimulateJobs } from './api';
import { ExportDialog } from './ExportDialog';
import { useHost } from './host';
import { NewRunPanel } from './NewRunPanel';
import { getRunRoute } from './paths';
import type { Job } from './types';

/** "1 run", "3 runs": a count with the noun it counts. */
const countOf = (count: number, noun: string) => `${count} ${noun}${count === 1 ? '' : 's'}`;

const targetName = (job: Job): string => job.spec?.target?.agent ?? job.spec?.target?.model ?? '';

/** Each multi-select filter's ticked values, keyed by column id. */
type Ticked = Record<string, Record<string, boolean> | undefined>;

const options = (values: Iterable<string>) =>
  [...new Set(values)].sort().map((value) => ({ value, label: value }));

/** What each sortable column sorts by. */
const SORT_KEYS: Record<string, (job: Job) => string | number> = {
  name: (job) => job.name,
  target: targetName,
  users: (job) => job.spec?.users ?? -1,
  status: (job) => job.status ?? '',
  created: (job) => job.created_at ?? '',
};

const makeColumns =
  (jobs: Job[]): ComponentProps<typeof StudioDataView<Job>>['makeColumns'] =>
  (col, { rowSelectionColumn }) => [
    rowSelectionColumn({ size: 48 }),
    col.accessor('name', { header: 'Run', size: 220, enableSorting: true }),
    col.accessor(targetName, {
      id: 'target',
      header: 'Tested',
      size: 200,
      enableSorting: true,
      meta: {
        filter: {
          type: 'multi-select',
          label: 'Tested',
          options: options(jobs.map(targetName)),
        },
      },
      cell: ({ row }) => targetName(row.original),
    }),
    col.accessor((job) => (job.spec?.probes ?? []).join(', '), {
      id: 'probes',
      header: 'Probes',
      size: 260,
      enableSorting: false,
      meta: {
        filter: {
          type: 'multi-select',
          label: 'Probes',
          options: options(jobs.flatMap((job) => job.spec?.probes ?? [])),
        },
      },
      cell: ({ row }) => row.original.spec.probes?.join(', '),
    }),
    col.accessor((job) => job.spec?.users, {
      id: 'users',
      header: 'Conversations',
      size: 140,
      enableSorting: true,
      cell: ({ row }) => row.original.spec.users,
    }),
    col.accessor((job) => job.status ?? '', {
      id: 'status',
      header: 'Status',
      size: 120,
      enableSorting: true,
      meta: {
        filter: {
          type: 'multi-select',
          label: 'Status',
          options: options(jobs.flatMap((job) => (job.status ? [job.status] : []))),
        },
      },
      cell: ({ row }) => <StatusBadge status={row.original.status} />,
    }),
    col.accessor((job) => job.created_at ?? '', {
      id: 'created',
      header: 'Started',
      size: 140,
      enableSorting: true,
      cell: ({ row }) =>
        row.original.created_at ? <RelativeTime datetime={row.original.created_at} /> : null,
    }),
  ];

/** Search, filters and sorting run here: the list holds every run, so Studio's table leaves them to us. */
const useVisibleRuns = (jobs: Job[], state: ReturnType<typeof useStudioDataViewState>) => {
  const { debouncedSearchBar, debouncedColumnFilters } = state;
  const filtered = useMemo(() => {
    const search = debouncedSearchBar.trim().toLowerCase();
    const ticked: Ticked = Object.fromEntries(
      debouncedColumnFilters.map((filter) => [filter.id, filter.value as Record<string, boolean>])
    );
    const allows = (id: string, values: string[]) => {
      const chosen = Object.keys(ticked[id] ?? {}).filter((value) => ticked[id]?.[value]);
      return !chosen.length || values.some((value) => chosen.includes(value));
    };
    return jobs.filter(
      (job) =>
        (!search || `${job.name} ${targetName(job)}`.toLowerCase().includes(search)) &&
        allows('target', [targetName(job)]) &&
        allows('probes', job.spec?.probes ?? []) &&
        allows('status', job.status ? [job.status] : [])
    );
  }, [jobs, debouncedSearchBar, debouncedColumnFilters]);

  const sort = state.sorting.state[0];
  const sorted = useMemo(() => {
    const key = sort && SORT_KEYS[sort.id];
    if (!key) return filtered;
    return [...filtered].sort((a, b) => {
      const [x, y] = [key(a), key(b)];
      const cmp =
        typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y));
      return sort.desc ? -cmp : cmp;
    });
  }, [filtered, sort]);

  const { pageIndex, pageSize } = state.pagination.state;
  return {
    page: useMemo(
      () => sorted.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
      [sorted, pageIndex, pageSize]
    ),
    total: filtered.length,
    narrowed: !!debouncedSearchBar.trim() || debouncedColumnFilters.length > 0,
  };
};

export const RunListRoute = () => {
  const { workspaceId, breadcrumbs, notifications } = useHost();
  const navigate = useNavigate();
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created', desc: true }],
  });
  const { data: jobs = [], isPending, isError } = useSimulateJobs();
  const runs = useVisibleRuns(jobs, dataViewState);
  const columns = useMemo(() => makeColumns(jobs), [jobs]);
  const [creating, setCreating] = useState(false);
  const [exporting, setExporting] = useState<string[] | null>(null);
  const [deleting, setDeleting] = useState<string[] | null>(null);
  const deleteRuns = useDeleteRuns();
  const cancelRuns = useCancelRuns();

  useEffect(() => {
    breadcrumbs.set([{ label: 'User Simulation' }]);
    return () => breadcrumbs.set([]);
  }, [breadcrumbs]);

  return (
    <Stack gap="density-lg" className="p-4">
      <Flex justify="between" align="center">
        <Stack gap="density-xs">
          <Text kind="body/semibold/lg">User Simulation</Text>
          <Text kind="body/regular/sm" className="text-subtle">
            Simulated users picked from census-grounded personas talk to your agent. Each
            conversation is scored and published to Intake for review.
          </Text>
        </Stack>
        <Button onClick={() => setCreating(true)}>New simulation</Button>
      </Flex>
      {isError ? (
        <TableEmptyState
          header="Couldn't load runs"
          emptyMessage="Check that the user-sim plugin service is running."
        />
      ) : (
        <StudioDataView<Job>
          dataViewState={dataViewState}
          makeColumns={columns}
          searchField="name"
          onRowClick={(job) => navigate(getRunRoute(workspaceId, job.name))}
          renderBulkActions={({ selectedRows, table }) => {
            const completed = selectedRows.filter((job) => job.status === 'completed');
            const running = selectedRows.filter((job) => !isFinished(job.status));
            const names = (runs: Job[]) => runs.map((job) => job.name);
            return (
              <>
                {completed.length ? (
                  <Button kind="tertiary" onClick={() => setExporting(names(completed))}>
                    Export {countOf(completed.length, 'completed run')}
                  </Button>
                ) : null}
                {running.length ? (
                  <Button
                    kind="tertiary"
                    onClick={() =>
                      cancelRuns.mutate(names(running), {
                        onSuccess: () => {
                          table.resetRowSelection();
                          notifications.notify(
                            `Stopping ${countOf(running.length, 'run')}.`,
                            'success'
                          );
                        },
                        onError: (error) => notifications.notify(error.message, 'error'),
                      })
                    }
                  >
                    Stop {countOf(running.length, 'running run')}
                  </Button>
                ) : null}
                <Button
                  kind="tertiary"
                  color="danger"
                  onClick={() => setDeleting(names(selectedRows))}
                >
                  Delete {countOf(selectedRows.length, 'run')}
                </Button>
              </>
            );
          }}
          attributes={{
            DataViewSearchBar: { placeholder: 'Search runs…' },
            DataViewRoot: {
              data: runs.page,
              totalCount: runs.total,
              requestStatus: isPending ? 'loading' : undefined,
              reactTableOptions: { getRowId: (job) => job.name },
            },
            DataViewTableContent: {
              renderEmptyState: () =>
                runs.narrowed ? (
                  <TableEmptyState
                    header="No matching simulations"
                    emptyMessage="Change the search or filters."
                  />
                ) : (
                  <TableEmptyState
                    header="No simulations yet"
                    emptyMessage="Start one here, or run nemo user-sim run from the CLI."
                  />
                ),
            },
          }}
        />
      )}
      <NewRunPanel open={creating} onClose={() => setCreating(false)} />
      {exporting ? <ExportDialog jobs={exporting} open onClose={() => setExporting(null)} /> : null}
      <DeleteConfirmationModal
        open={!!deleting}
        onClose={() => setDeleting(null)}
        onNotify={notifications.notify}
        simpleConfirm
        title={`Delete ${deleting?.length ?? 0} simulation run(s)?`}
        description="The runs and their results are deleted. Their conversations stay in Traces, and training data you already exported is kept."
        successText="Runs deleted."
        errorText="Some runs couldn't be deleted."
        onDelete={async () => {
          if (!deleting) return false;
          await deleteRuns.mutateAsync(deleting);
          return true;
        }}
      />
    </Stack>
  );
};
