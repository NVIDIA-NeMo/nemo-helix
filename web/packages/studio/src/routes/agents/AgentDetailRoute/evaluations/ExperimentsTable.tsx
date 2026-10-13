// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ROW_SELECTION_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import {
  deleteExperiment,
  getListExperimentsQueryKey,
} from '@nemo/sdk/generated/platform/experiments';
import { Button, Text } from '@nvidia/foundations-react-core';
import { BulkDeleteModal } from '@studio/components/BulkDeleteModal';
import type { AgentExperimentRow } from '@studio/routes/agents/AgentDetailRoute/evaluations/groupByExperiment';
import { getExperimentDetailRoute } from '@studio/routes/utils';
import { useQueryClient } from '@tanstack/react-query';
import { FolderTree, Trash } from 'lucide-react';
import { type ComponentProps, type FC, useCallback, useMemo, useState } from 'react';

interface ExperimentsTableProps {
  workspace: string;
  experiments: AgentExperimentRow[];
}

/** Deleting an experiment cascades to the evaluations whose only membership was that group, so the
 *  confirmation says how many go with it. */
const deleteTitle = (rows: AgentExperimentRow[]): string => {
  const evaluations = rows.reduce((total, row) => total + row.evaluationCount, 0);
  const experiments = `${rows.length} Experiment${rows.length !== 1 ? 's' : ''}`;
  return `Delete ${experiments} and ${evaluations} Evaluation${evaluations !== 1 ? 's' : ''}`;
};

/** The agent's evaluations rolled up by the experiment they belong to. Selecting one opens the
 *  experiment's own route, which already lists the evaluations under it. */
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
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: getListExperimentsQueryKey(workspace) }),
        queryClient.invalidateQueries({ queryKey: getListEvaluationsQueryKey(workspace) }),
      ]);

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
            renderEmptyState: () => (
              <TableEmptyState
                className="py-density-3xl"
                icon={<FolderTree className="size-16" />}
                header="No experiments yet"
                emptyMessage="An experiment appears here once one of its evaluations publishes results for this agent."
              />
            ),
          },
        }}
      />

      <BulkDeleteModal
        items={deleteRows}
        open={deleteRows.length > 0}
        onDelete={handleDelete}
        title={() => deleteTitle(deleteRows)}
        onClose={() => {
          setDeleteRows([]);
          dataViewState.rowSelection.set({});
        }}
      />
    </>
  );
};
