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
  matchesTaskFilters,
  parseTaskRef,
  requestStatus,
  RUNNER_LABELS,
  RUNNER_OPTIONS,
} from '@studio/routes/evaluation/tasksets/helpers';
import type { TaskFilters, TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useResolvedTasks } from '@studio/routes/evaluation/tasksets/useResolvedTasks';
import { type ComponentProps, type FC, useMemo } from 'react';

interface TaskTableProps {
  taskset: Taskset;
  onTaskClick: (row: TasksetTaskRow) => void;
}

/** The taskset's member tasks. Every member is resolved (in batches) so search and filters span
 *  all pages; paging is client-side since the refs are inline on the taskset. */
export const TaskTable: FC<TaskTableProps> = ({ taskset, onTaskClick }) => {
  const dataViewState = useStudioDataViewState<TaskFilters>({ defaultSort: [] });
  const { searchText, filter } = dataViewState.apiFilter;
  const { pageIndex, pageSize } = dataViewState.pagination.state;

  const members = useMemo(
    () =>
      (taskset.tasks ?? []).map(
        (ref): TasksetTaskRow => ({ id: ref, ref, ...parseTaskRef(ref, taskset.workspace) })
      ),
    [taskset]
  );
  const { rows, isFetching, error } = useResolvedTasks(members);
  const matching = rows.filter((row) => matchesTaskFilters(row, { ...filter, name: searchText }));

  const makeColumns: ComponentProps<typeof StudioDataView<TasksetTaskRow>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('name', {
      header: 'Name',
      size: 240,
      enableSorting: false,
      cell: ({ row }) => {
        const { name, workspace } = row.original;
        return <Text>{workspace === taskset.workspace ? name : `${workspace}/${name}`}</Text>;
      },
    }),
    accessor('version', {
      header: 'Revision',
      size: 150,
      enableSorting: false,
      cell: ({ row }) => {
        const { version } = row.original;
        if (!version) return <Text className="text-subtle">latest</Text>;
        return (
          <Text kind="mono/sm" title={version}>
            {version.slice(0, 12)}
          </Text>
        );
      },
    }),
    accessor((row) => row.task?.spec.kind ?? '', {
      id: 'runner',
      header: 'Runner',
      size: 110,
      enableSorting: false,
      meta: { filter: { type: 'multi-select', options: RUNNER_OPTIONS } },
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
      meta: { filter: { type: 'text', placeholder: 'key or value' } },
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
          data: matching.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize),
          totalCount: matching.length,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) => (
            <Text className="text-subtle p-density-xl">
              {hasFiltersApplied || hasSearchApplied
                ? 'No tasks match your search or filters.'
                : 'This taskset has no tasks.'}
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
