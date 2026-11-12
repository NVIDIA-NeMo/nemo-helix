// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useEvalsListTasks } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import type { Task, TaskFilter, TaskSort } from '@nemo/sdk/generated/evals/schema';
import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import type { TaskListFilters } from '@studio/routes/evaluation/tasks/types';
import {
  requestStatus,
  RUNNER_LABELS,
  RUNNER_OPTIONS,
  tagNames,
  taskSummary,
} from '@studio/routes/evaluation/tasksets/helpers';
import { getTaskDetailRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import type { ComponentProps, FC } from 'react';
import { useNavigate } from 'react-router';

export const TasksDataView: FC<{ workspace: string }> = ({ workspace }) => {
  const navigate = useNavigate();
  const dataViewState = useStudioDataViewState<TaskListFilters>({
    defaultSort: [{ id: 'updated_at', desc: true }],
  });
  const { searchText, filter: columns } = dataViewState.apiFilter;
  const kinds = Object.keys(columns?.runner ?? {});
  const filter = withOperators<TaskFilter>({
    ...(searchText && { name: { $like: `%${searchText}%` } }),
    ...(kinds.length && { kind: { $in: kinds } }),
    ...(columns?.tags && { tags: columns.tags }),
    ...(columns?.updated_at && { updated_at: columns.updated_at }),
  });

  const { data, isFetching, error } = useEvalsListTasks(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParam(dataViewState.sorting.state) as TaskSort,
      filter: Object.keys(filter).length ? filter : undefined,
    },
    { query: { placeholderData: keepPreviousData } }
  );

  // Loading placeholder rows carry no `spec`, so nested reads go through `spec?.`.
  const makeColumns: ComponentProps<typeof StudioDataView<Task>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('name', { header: 'Name', size: 200 }),
    accessor((row) => row.spec?.kind ?? '', {
      id: 'runner',
      header: 'Runner',
      size: 110,
      enableSorting: false,
      meta: { filter: { type: 'multi-select', options: RUNNER_OPTIONS } },
      cell: ({ row }) =>
        row.original.spec && (
          <Tag kind="outline" color="gray" readOnly>
            {RUNNER_LABELS[row.original.spec.kind]}
          </Tag>
        ),
    }),
    accessor((row) => (row.spec && taskSummary(row)) ?? '', {
      id: 'summary',
      header: 'Instruction',
      enableSorting: false,
      cell: ({ row }) => {
        const summary = row.original.spec && taskSummary(row.original);
        return (
          <Text className="truncate" title={summary}>
            {summary || EMPTY_FIELD_VALUE}
          </Text>
        );
      },
    }),
    accessor('revision', {
      header: 'Latest Revision',
      size: 130,
      enableSorting: false,
      cell: ({ row }) => <Text>r{row.original.revision}</Text>,
    }),
    accessor((row) => tagNames(row.tags).join(','), {
      id: 'tags',
      header: 'Tags',
      size: 180,
      enableSorting: false,
      meta: { filter: { type: 'text', label: 'Tag', placeholder: 'e.g. stable' } },
      cell: ({ row }) => (
        <Flex gap="density-xs" wrap="wrap">
          {tagNames(row.original.tags).map((tag) => (
            <Tag key={tag} kind="outline" color="gray" readOnly>
              {tag}
            </Tag>
          ))}
        </Flex>
      ),
    }),
    accessor('updated_at', {
      header: 'Updated',
      size: 140,
      meta: { filter: dateTimeFilter('Updated At') },
      cell: ({ row }) => <RelativeTime datetime={row.original.updated_at} />,
    }),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={(task) => navigate(getTaskDetailRoute(workspace, task.name))}
      attributes={{
        DataViewSearchBar: { placeholder: 'Search tasks...' },
        DataViewRoot: {
          data: data?.data ?? [],
          totalCount: data?.pagination?.total_results,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="tasks"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="tasks" variant="first-use" />
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
