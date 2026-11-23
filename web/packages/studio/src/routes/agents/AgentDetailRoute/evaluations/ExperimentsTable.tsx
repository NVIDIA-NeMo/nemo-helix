// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import {
  EntityEmptyState,
  type EntityEmptyStateProps,
} from '@nemo/common/src/components/EntityEmptyState';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Trash } from '@nemo/icons/icons';
import { deleteExperiment } from '@nemo/sdk/generated/platform/experiments';
import { Button, Text } from '@nvidia/foundations-react-core';
import { invalidateExperimentDeletion } from '@studio/api/evaluation/invalidateExperimentDeletion';
import { EXPERIMENT_DELETE_DESCRIPTION } from '@studio/components/ExperimentDeleteButton';
import type { AgentExperimentRow } from '@studio/routes/agents/AgentDetailRoute/evaluations/groupByExperiment';
import { getExperimentDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

interface ExperimentsTableProps {
  workspace: string;
  experiments: AgentExperimentRow[];
}

/** The agent's evaluations rolled up by the experiment they belong to. Selecting one opens the
 *  experiment's own route, which already lists the evaluations under it. */
/** Variant-specific `EntityEmptyState` props for the `experiments` entity, minus `entity`/`workspace`. */
type AgentExperimentsEmptyStateProps =
  | Omit<Extract<EntityEmptyStateProps, { variant: 'first-use' }>, 'entity' | 'workspace'>
  | Omit<Extract<EntityEmptyStateProps, { variant: 'no-results' }>, 'entity' | 'workspace'>;

export const ExperimentsTable: FC<ExperimentsTableProps> = ({ workspace, experiments }) => {
  const openRow = useRowNavigation();
  const queryClient = useQueryClient();
  const dataViewState = useStudioDataViewState();
  // StudioDataView renders exactly the rows it is given and never slices, so apply the page here.
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const pageRows = useMemo(
    () => experiments.slice(pageIndex * pageSize, pageIndex * pageSize + pageSize),
    [experiments, pageIndex, pageSize]
  );
  const [deleteRows, setDeleteRows] = useState<AgentExperimentRow[]>([]);

  const handleDelete = useCallback(
    async (rows: AgentExperimentRow[]) => {
      const names = rows.flatMap((row) => (row.name ? [row.name] : []));
      if (names.length !== rows.length) {
        throw new Error(
          'Some selected experiments could not be resolved to a name and cannot be deleted. Open the experiment to remove it.'
        );
      }

      const results = await Promise.allSettled(
        names.map((name) => deleteExperiment(workspace, name))
      );
      await invalidateExperimentDeletion(queryClient, workspace);

      const failed = rows.filter((_, index) => results[index]?.status === 'rejected');
      if (failed.length) {
        setDeleteRows(failed);
        throw new Error(
          `${failed.length} of ${rows.length} experiment${rows.length !== 1 ? 's' : ''} could not be deleted. Retry to attempt only those.`
        );
      }
    },
    [workspace, queryClient]
  );

  const makeColumns: ComponentProps<typeof StudioDataView<AgentExperimentRow>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowSelectionColumn }) => [
        rowSelectionColumn({ size: ROW_SELECTION_COLUMN_SIZE }),
        accessor('name', {
          header: 'Experiment',
          cell: ({ row }) => (
            <Text title={row.original.name ?? row.original.id}>
              {row.original.name ?? row.original.id}
            </Text>
          ),
        }),
        accessor('evaluationCount', {
          header: 'Evaluations',
          cell: ({ row }) => <Text>{row.original.evaluationCount}</Text>,
        }),
        accessor('runCount', {
          header: 'Runs',
          cell: ({ row }) => <Text>{row.original.runCount}</Text>,
        }),
        accessor('latestCreatedAt', {
          header: 'Latest run',
          cell: ({ row }) =>
            row.original.latestCreatedAt ? (
              <RelativeTime datetime={row.original.latestCreatedAt} />
            ) : (
              '—'
            ),
        }),
      ],
      []
    );

  return (
    <>
      <StudioDataView<AgentExperimentRow>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row, _index, event) =>
          row.name && openRow(event, getExperimentDetailRoute(workspace, row.name))
        }
        renderBulkActions={({ selectedRows }) => (
          <Button
            kind="tertiary"
            aria-label="Delete selected experiments"
            onClick={() => setDeleteRows(selectedRows)}
          >
            <Trash /> Delete
          </Button>
        )}
        attributes={{
          DataViewRoot: { data: pageRows, totalCount: experiments.length },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) => {
              const emptyStateProps: AgentExperimentsEmptyStateProps =
                hasFiltersApplied || hasSearchApplied
                  ? { variant: 'no-results', onClearFilters: dataViewState.resetFilters }
                  : { variant: 'first-use' };
              return (
                <EntityEmptyState entity="experiments" workspace={workspace} {...emptyStateProps} />
              );
            },
          },
        }}
      />

      {deleteRows.length > 0 && (
        <DeleteConfirmationModal
          open
          entity="experiment"
          ids={deleteRows.map((row) => row.name ?? 'unnamed')}
          description={EXPERIMENT_DELETE_DESCRIPTION}
          onDelete={async () => {
            await handleDelete(deleteRows);
            return true;
          }}
          onClose={() => {
            setDeleteRows([]);
            dataViewState.rowSelection.set({});
          }}
        />
      )}
    </>
  );
};
