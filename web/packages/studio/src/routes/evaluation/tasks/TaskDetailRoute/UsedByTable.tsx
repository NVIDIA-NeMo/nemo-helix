// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Text } from '@nvidia/foundations-react-core';
import { PinnedRevision } from '@studio/routes/evaluation/tasks/TaskDetailRoute/PinnedRevision';
import { useTaskUsage } from '@studio/routes/evaluation/tasks/TaskDetailRoute/useTaskUsage';
import type { TaskUsageRow } from '@studio/routes/evaluation/tasks/types';
import { requestStatus } from '@studio/routes/evaluation/tasksets/helpers';
import { getTasksetDetailRoute } from '@studio/routes/utils';
import type { ComponentProps, FC } from 'react';
import { useNavigate } from 'react-router';

/** Tasksets in the task's workspace that have it as a member, with the revision each pins. */
export const UsedByTable: FC<{ task: Task }> = ({ task }) => {
  const navigate = useNavigate();
  const dataViewState = useStudioDataViewState({ defaultSort: [] });
  const { ref, rows, totalCount, isFetching, error } = useTaskUsage(task, {
    page: dataViewState.pagination.state.pageIndex + 1,
    page_size: dataViewState.pagination.state.pageSize,
  });

  const makeColumns: ComponentProps<typeof StudioDataView<TaskUsageRow>>['makeColumns'] = ({
    accessor,
  }) => [
    accessor('name', { header: 'Taskset', size: 220, enableSorting: false }),
    accessor('pinnedRevision', {
      header: 'Pinned Task Revision',
      size: 220,
      enableSorting: false,
      cell: ({ row }) => <PinnedRevision {...row.original} />,
    }),
    accessor('revision', {
      header: 'Taskset Revision',
      size: 140,
      enableSorting: false,
      cell: ({ row }) => <Text>r{row.original.revision}</Text>,
    }),
    accessor('updated_at', {
      header: 'Updated',
      size: 140,
      enableSorting: false,
      cell: ({ row }) => <RelativeTime datetime={row.original.updated_at} />,
    }),
  ];

  return (
    <StudioDataView
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      onRowClick={(taskset) =>
        navigate(getTasksetDetailRoute(taskset.workspace, taskset.name, { task: ref }))
      }
      attributes={{
        DataViewRoot: {
          data: rows,
          totalCount,
          requestStatus: requestStatus(error, isFetching),
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <Text className="text-subtle p-density-xl">No taskset uses this task.</Text>
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
