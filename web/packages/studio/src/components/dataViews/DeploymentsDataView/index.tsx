/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { withOperators } from '@nemo/common/src/api/filterOperators';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { EntityEmptyState } from '@nemo/common/src/components/EntityEmptyState';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { useModelsListDeployments } from '@nemo/sdk/generated/platform/model-deployments';
import {
  ModelDeployment,
  ModelDeploymentFilter,
  ModelDeploymentStatus,
} from '@nemo/sdk/generated/platform/schema';
import { type DropdownEntry, Stack, Text } from '@nvidia/foundations-react-core';
import { keepPreviousData } from '@tanstack/react-query';
import { type ComponentProps, type FC, useCallback } from 'react';

export interface DeploymentsDataViewProps {
  workspace: string;
  /** Opens the create-deployment flow from the first-use empty state. */
  readonly onCreate?: () => void;
  /** Opens the URL-driven deployment details panel (row click). */
  onDeploymentRowClick: (deployment: ModelDeployment) => void;
  /** Opens the shared delete confirmation flow (row action menu). */
  onRequestDeleteDeployment: (deployment: ModelDeployment) => void;
  attributes?: {
    Stack?: React.ComponentProps<typeof Stack>;
  };
}

export const DeploymentsDataView: FC<DeploymentsDataViewProps> = ({
  workspace,
  onCreate,
  onDeploymentRowClick,
  onRequestDeleteDeployment,
  attributes,
}) => {
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
  });

  const sortState = dataViewState.sorting.state[0];
  const sortParam = sortState ? `${sortState.desc ? '-' : ''}${sortState.id}` : '-created_at';

  const { data, isFetching, error } = useModelsListDeployments(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: sortParam,
      filter: {
        ...dataViewState.apiFilter.filter,
        ...(dataViewState.apiFilter.searchText
          ? withOperators<ModelDeploymentFilter>({
              name: { $like: dataViewState.apiFilter.searchText },
            })
          : {}),
      },
    },
    {
      query: {
        placeholderData: keepPreviousData,
      },
    }
  );

  const pagination = data?.pagination;

  const makeColumns: ComponentProps<typeof StudioDataView<ModelDeployment>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowActionsColumn }) => [
        accessor('name', {
          header: 'Name',
          enableSorting: false,
          size: 175,
          cell({ row }) {
            return <Text className="font-bold">{row.original.name}</Text>;
          },
        }),
        accessor('status', {
          header: 'Status',
          size: 120,
          cell({ row }) {
            return <StatusBadge status={row.original.status} />;
          },
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 150,
          cell({ row }) {
            return row.original.created_at ? (
              <RelativeTime datetime={row.original.created_at} />
            ) : (
              <Text>-</Text>
            );
          },
        }),
        rowActionsColumn({
          size: 58,
          enableResizing: false,
          rowActions: (deployment: ModelDeployment): DropdownEntry[] => [
            {
              children: 'Delete',
              disabled:
                deployment.status === ModelDeploymentStatus.DELETED ||
                deployment.status === ModelDeploymentStatus.DELETING,
              danger: true,
              onSelect: () => onRequestDeleteDeployment(deployment),
            },
          ],
        }),
      ],
      [onRequestDeleteDeployment]
    );

  return (
    <Stack gap="density-2xl" {...attributes?.Stack}>
      <StudioDataView
        dataViewState={dataViewState}
        searchField="name"
        makeColumns={makeColumns}
        onRowClick={onDeploymentRowClick}
        attributes={{
          DataViewSearchBar: {
            placeholder: 'Search Deployments...',
          },
          DataViewRoot: {
            data: data?.data ?? [],
            totalCount: pagination?.total_results,
            requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
          },
          DataViewTableContent: {
            renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
              hasFiltersApplied || hasSearchApplied ? (
                <EntityEmptyState
                  entity="deployments"
                  variant="no-results"
                  onClearFilters={dataViewState.resetFilters}
                />
              ) : (
                <EntityEmptyState entity="deployments" variant="first-use" onCreate={onCreate} />
              ),
            renderErrorState: () => (
              <ErrorPanel
                errorMessage={getErrorMessage(error ?? new Error('Failed to fetch deployments'))}
              />
            ),
          },
        }}
      />
    </Stack>
  );
};
