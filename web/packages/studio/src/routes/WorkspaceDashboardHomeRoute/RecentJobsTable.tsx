// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import type { RowClickHandler } from '@nemo/common/src/components/DataView/useRowClick';
import { ElapsedTime } from '@nemo/common/src/components/ElapsedTime';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useRowNavigation } from '@nemo/common/src/hooks/useRowNavigation';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import type { HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { Flex, Tag } from '@nvidia/foundations-react-core';
import {
  getSourceDisplay,
  getSourceTitle,
} from '@studio/components/dataViews/JobsDataView/sourceDisplay';
import { getJobDetailRoute } from '@studio/components/dataViews/JobsDataView/utils';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getNewCustomizationJobRoute } from '@studio/routes/utils';
import { isJobRunning } from '@studio/routes/WorkspaceDashboardHomeRoute/recentJobsUtils';
import { useRecentJobs } from '@studio/routes/WorkspaceDashboardHomeRoute/useRecentJobs';
import { type ComponentProps, useCallback } from 'react';
import { useNavigate } from 'react-router';

// Module-level, with a stable onRowClick, so cells don't remount on every poll.
const makeColumns: ComponentProps<typeof StudioDataView<HelixJobResponse>>['makeColumns'] = ({
  accessor,
}) => [
  accessor((original) => original?.name || '', {
    id: 'name',
    header: 'Name',
  }),
  accessor((original) => original?.source || '', {
    id: 'source',
    header: 'Source',
    meta: {
      title: (cell) => getSourceTitle(cell.getValue()),
    },
    cell: ({ row }) => {
      const sourceValue = row.original.source;
      if (!sourceValue) {
        return '-';
      }
      const { icon, label } = getSourceDisplay(sourceValue);
      return (
        <Tag kind="outline" color="gray" density="compact" readOnly>
          <Flex align="center" gap="density-sm">
            {icon}
            {label}
          </Flex>
        </Tag>
      );
    },
  }),
  accessor((original) => original?.status || '', {
    id: 'status',
    header: 'Status',
    cell: ({ row }) => <StatusBadge status={row.original.status} />,
  }),
  accessor((original) => original?.created_at || '', {
    id: 'elapsed',
    header: 'Elapsed time',
    meta: { title: false },
    cell: ({ row }) => (
      <ElapsedTime
        start={row.original.created_at}
        end={row.original.updated_at}
        running={isJobRunning(row.original.status)}
      />
    ),
  }),
];

export const RecentJobsTable = () => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const openRow = useRowNavigation();

  const dataViewState = useStudioDataViewState({
    defaultPageSize: 10,
  });
  const { pageRows, totalCount, isLoading, error } = useRecentJobs(workspace, dataViewState);

  const handleRowClick = useCallback<RowClickHandler<HelixJobResponse>>(
    (row, _index, event) => openRow(event, getJobDetailRoute(row, workspace)),
    [openRow, workspace]
  );

  if (error) {
    return <ErrorPanel errorMessage={getErrorMessage(error)} />;
  }

  return (
    <StudioDataView<HelixJobResponse>
      dataViewState={dataViewState}
      makeColumns={makeColumns}
      onRowClick={handleRowClick}
      attributes={{
        DataViewRoot: {
          data: pageRows,
          totalCount,
          requestStatus: isLoading ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: () => (
            <EntityEmptyState
              entity="recentJobs"
              className="py-density-xl"
              variant="first-use"
              onCreate={
                CUSTOMIZER_ENABLED
                  ? () => navigate(getNewCustomizationJobRoute(workspace))
                  : undefined
              }
            />
          ),
        },
      }}
    />
  );
};
