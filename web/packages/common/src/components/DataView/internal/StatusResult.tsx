// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { IconError, Filter, Refresh } from '@nemo/icons/icons';
import {
  Button,
  Flex,
  StatusMessage,
  type StatusMessageProps,
} from '@nvidia/foundations-react-core';
import type { JSX } from 'react';

export type TableStatusState = Pick<
  ReturnType<typeof useInnerDataViewContext>,
  'isDataViewEmptyState' | 'isDataViewErrorState' | 'table' | 'state'
>;

export interface StatusResultProps extends Partial<StatusMessageProps> {
  /** Render a custom error state. */
  renderErrorState?: (tableState: TableStatusState) => JSX.Element;
  /** Render a custom empty state. */
  renderEmptyState?: (
    tableState: TableStatusState & {
      hasFiltersApplied: boolean;
      hasSearchApplied: boolean;
    }
  ) => JSX.Element | null;
}

export function StatusResult({
  renderErrorState,
  renderEmptyState,
  ...props
}: StatusResultProps): JSX.Element | null {
  const { isDataViewEmptyState, isDataViewErrorState, table, state } = useInnerDataViewContext();
  if (isDataViewErrorState) {
    if (renderErrorState) {
      return renderErrorState({ isDataViewEmptyState, isDataViewErrorState, table, state });
    }
    return (
      <Flex justify="center" className="p-density-md">
        <StatusMessage
          attributes={{ StatusMessageMedia: { className: '!text-feedback-danger' } }}
          slotMedia={<IconError variant="fill" />}
          slotHeading="Something went wrong"
          slotSubheading="There was an error loading the data. If the problem persists, please contact support."
          slotFooter={
            <Button onClick={() => window.location.reload()} kind="secondary" size="small">
              <Refresh />
            </Button>
          }
          {...props}
        />
      </Flex>
    );
  }
  if (isDataViewEmptyState) {
    const tableState = table.getState();
    const hasFiltersApplied = tableState.columnFilters.length > 0;
    const hasSearchApplied =
      typeof tableState.globalFilter === 'string'
        ? tableState.globalFilter.trim().length > 0
        : Boolean(tableState.globalFilter);
    const action =
      hasFiltersApplied || hasSearchApplied ? (
        <Button
          onClick={() => {
            if (hasFiltersApplied) {
              table.resetColumnFilters(true);
            }
            if (hasSearchApplied) {
              state.searchBar.set('');
            }
          }}
          kind="secondary"
          size="small"
        >
          Clear Filters
        </Button>
      ) : undefined;
    return renderEmptyState ? (
      renderEmptyState({
        isDataViewEmptyState,
        isDataViewErrorState,
        table,
        state,
        hasFiltersApplied,
        hasSearchApplied,
      })
    ) : (
      <Flex justify="center" className="p-density-md">
        <StatusMessage
          slotMedia={<Filter variant="line" />}
          slotHeading="No results"
          slotSubheading={
            hasFiltersApplied || hasSearchApplied
              ? 'No results match your filters'
              : 'There are no results to display.'
          }
          slotFooter={action}
          {...props}
        />
      </Flex>
    );
  }
  return null;
}
