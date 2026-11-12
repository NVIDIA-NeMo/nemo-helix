// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import {
  ROW_ACTIONS_COLUMN_SIZE,
  StudioDataView,
} from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { getSortParam } from '@nemo/common/src/utils/query';
import { useEvalsListTasksets } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import type { Taskset, TasksetFilter, TasksetSort } from '@nemo/sdk/generated/evals/schema';
import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { requestStatus, tagNames } from '@studio/routes/evaluation/tasksets/helpers';
import { TasksetActionsMenu } from '@studio/routes/evaluation/tasksets/TasksetActions';
import type { TasksetActionRequest } from '@studio/routes/evaluation/tasksets/types';
import { getTasksetDetailRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import type { ComponentProps, FC } from 'react';
import { useNavigate } from 'react-router';

interface Props {
  workspace: string;
  onCreate: () => void;
  onRequestAction: (request: TasksetActionRequest) => void;
}

export const TasksetsDataView: FC<Props> = ({ workspace, onCreate, onRequestAction }) => {
  const navigate = useNavigate();
  const dataViewState = useStudioDataViewState<TasksetFilter>({
    defaultSort: [{ id: 'updated_at', desc: true }],
  });
  const { searchText, filter: columns } = dataViewState.apiFilter;
  const filter = withOperators<TasksetFilter>({
    ...(searchText && { name: { $like: `%${searchText}%` } }),
    ...(columns?.description && { description: { $like: `%${columns.description}%` } }),
    ...(columns?.tags && { tags: columns.tags }),
    ...(columns?.updated_at && { updated_at: columns.updated_at }),
  });

  const { data, isFetching, error } = useEvalsListTasksets(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParam(dataViewState.sorting.state) as TasksetSort,
      filter: Object.keys(filter).length ? filter : undefined,
    },
    { query: { placeholderData: keepPreviousData } }
  );

  const makeColumns: ComponentProps<typeof StudioDataView<Taskset>>['makeColumns'] = (
    { accessor },
    { rowActionsColumn }
  ) => [
    accessor('name', { header: 'Name', size: 200 }),
    accessor('description', {
      header: 'Description',
      enableSorting: false,
      meta: { filter: { type: 'text', placeholder: 'Description contains' } },
      cell: ({ row }) => (
        <Text className="truncate" title={row.original.description}>
          {row.original.description || EMPTY_FIELD_VALUE}
        </Text>
      ),
    }),
    accessor((row) => row.tasks?.length ?? 0, {
      id: 'tasks',
      header: 'Tasks',
      size: 90,
      enableSorting: false,
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
    rowActionsColumn({
      size: ROW_ACTIONS_COLUMN_SIZE,
      enableResizing: false,
      cell: ({ row }) => <TasksetActionsMenu taskset={row.original} onRequest={onRequestAction} />,
    }),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      onRowClick={(taskset) => navigate(getTasksetDetailRoute(workspace, taskset.name))}
      attributes={{
        DataViewSearchBar: { placeholder: 'Search tasksets...' },
        DataViewRoot: {
          data: data?.data ?? [],
          totalCount: data?.pagination?.total_results,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="tasksets"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="tasksets" variant="first-use" onCreate={onCreate} />
            ),
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(error ?? new Error('Failed to fetch tasksets'))}
            />
          ),
        },
      }}
    />
  );
};
