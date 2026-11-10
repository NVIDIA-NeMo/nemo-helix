// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { Taskset } from '@nemo/sdk/generated/evals/schema';
import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import {
  formatValue,
  parseTaskRef,
  requestStatus,
  RUNNER_LABELS,
} from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useResolvedTasks } from '@studio/routes/evaluation/tasksets/useResolvedTasks';
import { type ComponentProps, type FC, useMemo } from 'react';

interface TaskTableProps {
  taskset: Taskset;
  onTaskClick: (row: TasksetTaskRow) => void;
}

/** The taskset's member tasks. Paged and searched client-side: refs are inline on the taskset,
 *  and only the visible page's task records are fetched. */
export const TaskTable: FC<TaskTableProps> = ({ taskset, onTaskClick }) => {
  const dataViewState = useStudioDataViewState({ defaultSort: [] });
  const search = dataViewState.searchBar.state.toLowerCase();
  const { pageIndex, pageSize } = dataViewState.pagination.state;

  const matching = useMemo(
    () =>
      (taskset.tasks ?? [])
        .map((ref): TasksetTaskRow => ({ id: ref, ref, ...parseTaskRef(ref, taskset.workspace) }))
        .filter((row) => !search || row.name.toLowerCase().includes(search)),
    [taskset, search]
  );
  const page = useMemo(
    () => matching.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
    [matching, pageIndex, pageSize]
  );
  const { rows, isFetching, error } = useResolvedTasks(page);

  const makeColumns: ComponentProps<typeof StudioDataView<TasksetTaskRow>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('name', {
      header: 'Name',
      size: 220,
      enableSorting: false,
      cell: ({ row }) => {
        const { name, workspace, version } = row.original;
        const qualified = workspace === taskset.workspace ? name : `${workspace}/${name}`;
        return (
          <Text className="truncate" title={row.original.ref}>
            {version ? `${qualified} @ ${version.slice(0, 12)}` : qualified}
          </Text>
        );
      },
    }),
    accessor((row) => row.task?.spec.kind ?? '', {
      id: 'runner',
      header: 'Runner',
      size: 110,
      enableSorting: false,
      cell: ({ row }) => {
        const kind = row.original.task?.spec.kind;
        if (kind)
          return (
            <Tag kind="outline" color="gray" readOnly>
              {RUNNER_LABELS[kind]}
            </Tag>
          );
        return <Text className="text-subtle">{isFetching ? EMPTY_FIELD_VALUE : 'Not found'}</Text>;
      },
    }),
    accessor((row) => row.task?.metadata?.length ?? 0, {
      id: 'metadata',
      header: 'Metadata',
      enableSorting: false,
      cell: ({ row }) => {
        const metadata = row.original.task?.metadata ?? [];
        if (metadata.length === 0) return <Text>{EMPTY_FIELD_VALUE}</Text>;
        return (
          <Flex gap="density-xs" wrap="wrap">
            {metadata.map(({ key, value }) => (
              <Tag key={key} kind="outline" color="gray" readOnly>
                {`${key}: ${formatValue(value)}`}
              </Tag>
            ))}
          </Flex>
        );
      },
    }),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={onTaskClick}
      attributes={{
        DataViewSearchBar: { placeholder: 'Search tasks...' },
        DataViewRoot: {
          data: rows,
          totalCount: matching.length,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <Text className="text-subtle p-density-xl">
              {search ? 'No tasks match your search.' : 'This taskset has no tasks.'}
            </Text>
          ),
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(error ?? new Error('Failed to fetch tasks'))}
            />
          ),
        },
      }}
    />
  );
};
