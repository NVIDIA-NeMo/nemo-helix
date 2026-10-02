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
import { useGuardrailsListGuardrailConfigs } from '@nemo/sdk/generated/platform/guardrails';
import type {
  GuardrailConfig,
  GuardrailConfigFilter,
  GuardrailsListGuardrailConfigsParams,
} from '@nemo/sdk/generated/platform/schema';
import { Badge, Button, type DropdownEntry, Flex, Text } from '@nvidia/foundations-react-core';
import {
  getMainModelName,
  getRailCounts,
} from '@studio/components/dataViews/GuardrailsDataView/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { Trash } from 'lucide-react';
import { type ComponentProps, type FC, useCallback } from 'react';

export interface GuardrailsDataViewProps {
  workspace: string;
  onRowClick: (config: GuardrailConfig, event: React.MouseEvent) => void;
  onRequestDuplicate?: (config: GuardrailConfig) => void;
  onRequestDelete?: (config: GuardrailConfig) => void;
  /** Opens the create-guardrail flow from the first-use empty state. */
  onCreate?: () => void;
  onRequestBulkDelete?: (configs: GuardrailConfig[]) => void;
}

export const GuardrailsDataView: FC<GuardrailsDataViewProps> = ({
  workspace,
  onRowClick,
  onRequestDuplicate,
  onRequestDelete,
  onCreate,
  onRequestBulkDelete,
}) => {
  const dataViewState = useStudioDataViewState({
    defaultSort: [{ id: 'created_at', desc: true }],
    columnVisibility: { created_at: false },
  });

  const { data, isFetching, error } = useGuardrailsListGuardrailConfigs(
    workspace,
    {
      page: dataViewState.pagination.state.pageIndex + 1,
      page_size: dataViewState.pagination.state.pageSize,
      sort: getSortParam(
        dataViewState.sorting.state
      ) as GuardrailsListGuardrailConfigsParams['sort'],
      filter: {
        ...((dataViewState.apiFilter.filter ?? {}) as GuardrailConfigFilter),
        ...(dataViewState.apiFilter.searchText
          ? withOperators<GuardrailConfigFilter>({
              name: { $like: dataViewState.apiFilter.searchText },
            })
          : {}),
      },
    },
    {
      query: { placeholderData: keepPreviousData },
    }
  );

  const pagination = data?.pagination;

  const makeColumns: ComponentProps<typeof StudioDataView<GuardrailConfig>>['makeColumns'] =
    useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn(),
        accessor('name', {
          header: 'Name',
          enableSorting: true,
          size: 180,
          cell({ row }) {
            return <Text className="font-bold">{row.original.name}</Text>;
          },
        }),
        accessor('data', {
          id: 'models',
          header: 'Main Model',
          enableSorting: false,
          cell({ row }) {
            const name = getMainModelName(row.original.data);
            return (
              <Text className="truncate" title={name}>
                {name ?? ''}
              </Text>
            );
          },
        }),
        accessor('data', {
          id: 'flows',
          header: 'Flows',
          enableSorting: false,
          size: 140,
          cell({ row }) {
            const { input, output } = getRailCounts(row.original.data);
            return (
              <Flex gap="1">
                {input > 0 && (
                  <Badge kind="solid" color="gray">
                    Input
                  </Badge>
                )}
                {output > 0 && (
                  <Badge kind="solid" color="gray">
                    Output
                  </Badge>
                )}
              </Flex>
            );
          },
        }),
        accessor('updated_at', {
          header: 'Updated',
          enableSorting: true,
          meta: {
            filter: dateTimeFilter('Updated At'),
          },
          cell({ row }) {
            return row.original.updated_at ? (
              <RelativeTime datetime={row.original.updated_at} />
            ) : (
              <Text>—</Text>
            );
          },
        }),
        accessor('created_at', {
          id: 'created_at',
          header: 'Created',
          enableSorting: true,
          meta: {
            filter: dateTimeFilter('Created At'),
          },
          cell({ row }) {
            return row.original.created_at ? (
              <RelativeTime datetime={row.original.created_at} />
            ) : (
              <Text>—</Text>
            );
          },
        }),
        rowActionsColumn({
          size: ROW_ACTIONS_COLUMN_SIZE,
          enableResizing: false,
          rowActions: (config: GuardrailConfig): DropdownEntry[] => [
            {
              children: 'Duplicate',
              onSelect: () => onRequestDuplicate?.(config),
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => onRequestDelete?.(config),
            },
          ],
        }),
      ],
      [onRequestDuplicate, onRequestDelete]
    );

  return (
    <StudioDataView
      dataViewState={dataViewState}
      searchField="name"
      makeColumns={makeColumns}
      renderBulkActions={({ selectedRows, table }) => (
        <Button
          kind="tertiary"
          aria-label="Delete selected guardrails"
          onClick={() => {
            onRequestBulkDelete?.(selectedRows);
            table.resetRowSelection();
          }}
        >
          <Trash /> Delete
        </Button>
      )}
      onRowClick={(row: GuardrailConfig, _index, event) => onRowClick(row, event)}
      attributes={{
        DataViewSearchBar: { placeholder: 'Search Guardrail Configs...' },
        DataViewRoot: {
          data: data?.data ?? [],
          totalCount: pagination?.total_results,
          requestStatus: error ? 'error' : isFetching ? 'loading' : undefined,
        },
        DataViewTableContent: {
          renderEmptyState: ({ hasFiltersApplied, hasSearchApplied }) =>
            hasFiltersApplied || hasSearchApplied ? (
              <EntityEmptyState
                entity="guardrails"
                variant="no-results"
                onClearFilters={dataViewState.resetFilters}
              />
            ) : (
              <EntityEmptyState entity="guardrails" variant="first-use" onCreate={onCreate} />
            ),
          renderErrorState: () => (
            <ErrorPanel
              errorMessage={getErrorMessage(
                error ?? new Error('Failed to fetch guardrail configs')
              )}
            />
          ),
        },
      }}
    />
  );
};
