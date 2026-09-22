// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { useRowClick } from '@nemo/common/src/components/DataView/useRowClick';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { DEFAULT_PAGE_SIZE_OPTIONS } from '@nemo/common/src/constants/pagination';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { Button, Text } from '@nvidia/foundations-react-core';
import { getExperimentDetailRoute } from '@studio/routes/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, type FC } from 'react';
import { useNavigate } from 'react-router';

const makeColumns: ComponentProps<typeof DataView.Root<ExperimentResponse>>['makeColumns'] = ({
  accessor,
}) => [
  accessor('name', {
    header: 'Experiments',
    enableSorting: false,
    size: 280,
    cell: ({ getValue }) => <Text>{getValue()}</Text>,
  }),
  accessor('evaluation_count', {
    header: 'Evaluations',
    enableSorting: false,
    size: 100,
    cell: ({ getValue }) => <Text>{getValue() ?? 0}</Text>,
  }),
  accessor('updated_at', {
    header: 'Updated',
    enableSorting: false,
    size: 120,
    cell: ({ row }) =>
      row.original.updated_at ? (
        <RelativeTime datetime={row.original.updated_at} />
      ) : (
        <Text>—</Text>
      ),
  }),
];

interface InsightExperimentsProps {
  workspace: string;
  insightId: string;
}

export const InsightExperiments: FC<InsightExperimentsProps> = ({ workspace, insightId }) => {
  const navigate = useNavigate();
  const dataViewState = DataView.useDataViewState({
    pagination: { paginationOptions: DEFAULT_PAGE_SIZE_OPTIONS },
  });
  const { pageIndex, pageSize } = dataViewState.pagination.state;
  const {
    data: response,
    isError,
    isFetching,
    refetch,
  } = useListExperiments(
    workspace,
    {
      page: pageIndex + 1,
      page_size: pageSize,
      sort: '-created_at',
      filter: { insight_id: insightId },
    },
    { query: { placeholderData: keepPreviousData } }
  );
  const groups = response?.data ?? [];
  const { wrapColumns, onClick, className } = useRowClick(
    (group: ExperimentResponse) => navigate(getExperimentDetailRoute(workspace, group.name)),
    groups
  );

  return (
    <DataView.Root
      className="min-h-[160px]"
      dataMode="manual"
      state={dataViewState}
      data={groups}
      totalCount={response?.pagination?.total_results ?? 0}
      requestStatus={isError ? 'error' : isFetching ? 'loading' : undefined}
      loadingRows={pageSize}
      makeColumns={wrapColumns(makeColumns)}
    >
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <DataView.TableContent
          className={`min-h-0 flex-1 overflow-auto bg-transparent [&_td]:!bg-transparent [&_thead]:!bg-transparent [&_thead_th]:!bg-transparent ${className}`}
          onClick={onClick}
          renderEmptyState={() => (
            <EntityEmptyState entity="insightExperiments" variant="first-use" />
          )}
          renderErrorState={() => (
            <ErrorMessage
              header="Failed to load experiments"
              message="The experiments for this insight could not be loaded."
              slotFooter={
                <Button type="button" kind="tertiary" onClick={() => void refetch()}>
                  Retry
                </Button>
              }
            />
          )}
        />
        <DataView.Pagination
          className="px-density-md py-density-sm"
          showItemsPerPage
          pageSizeOptions={DEFAULT_PAGE_SIZE_OPTIONS}
        />
      </div>
    </DataView.Root>
  );
};
