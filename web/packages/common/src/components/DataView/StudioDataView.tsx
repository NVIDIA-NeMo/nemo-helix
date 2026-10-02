// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnFilterPanel } from '@nemo/common/src/components/DataView/ColumnFilterPanel';
import { FilterPanel } from '@nemo/common/src/components/DataView/FilterPanel';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import '@nemo/common/src/components/DataView/StudioDataView.css';
import { StudioDataViewToolbar } from '@nemo/common/src/components/DataView/StudioDataViewToolbar';
import { RowClickHandler, useRowClick } from '@nemo/common/src/components/DataView/useRowClick';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { DEFAULT_PAGE_SIZE_OPTIONS } from '@nemo/common/src/constants/pagination';
import {
  PaginationArrowButton,
  PaginationDivider,
  PaginationControlsGroup,
  PaginationItemRangeText,
  PaginationNavigationGroup,
  PaginationPageCountText,
  PaginationPageInput,
  PaginationPageSizeSelect,
  Text,
  Flex,
  Block,
  Stack,
} from '@nvidia/foundations-react-core';
import { ComponentProps, ReactNode, RefObject, useMemo, useState } from 'react';

const PREBUILT_COLUMN_IDS = new Set(['row-selection', 'row-actions', 'row-expansion']);

export const ROW_SELECTION_COLUMN_SIZE = 50;
export const ROW_ACTIONS_COLUMN_SIZE = 50;

/**
 * Implements a common DataView component for Studio.
 * Opinionated prop choices for DataView components for consistency across Studio.
 */
interface Props<DataType> {
  dataViewState: DataView.DataViewState;
  makeColumns: ComponentProps<typeof DataView.Root<DataType>>['makeColumns'];
  /**
   * Called when a row is clicked or activated via keyboard (Enter/Space).
   * When provided, rows become clickable with cursor-pointer styling and keyboard-navigable.
   * Clicks on interactive child elements (buttons, links, inputs, etc.) are excluded automatically.
   * Add `data-no-row-click` to any element to opt it out of row-click delegation.
   */
  onRowClick?: RowClickHandler<DataType>;
  /**
   * Maximum number of text lines to show in each data cell before truncating
   * with an ellipsis. Prebuilt columns (row-selection, row-actions) are not affected.
   */
  maxTwoLines?: boolean;
  /**
   * The data field to search against. When provided, the search bar is rendered
   * in the toolbar. When omitted, no search bar is shown.
   */
  searchField?: string;
  /**
   * Render function for bulk actions shown in the toolbar when rows are selected.
   * Receives the selected rows' original data (unwrapped from TanStack Row objects).
   * When omitted, no bulk actions toolbar is rendered.
   */
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: DataView.TanstackTable.Table<DataType>;
  }) => ReactNode;
  /**
   * Custom content rendered in place of DataView.TableContent + DataView.Pagination.
   * Use this to render a card grid or other non-table layout inside DataView.Root.
   * When omitted, the default table with pagination is rendered.
   */
  children?: ReactNode;
  /** Rendered at the leading end of the toolbar row, before the search bar. */
  toolbarSlotStart?: ReactNode;
  /**
   * Rendered at the trailing end of the toolbar row (after the search bar and filter toggle).
   * Useful for controls like a sort dropdown that belong visually in the toolbar.
   */
  toolbarSlotEnd?: ReactNode;
  /**
   * Ref attached to the scrollable container that wraps custom `children`.
   * Pass this to a virtualizer so it can observe the scroll position.
   */
  scrollContainerRef?: RefObject<HTMLDivElement | null>;
  attributes?: {
    DataViewRoot?: Omit<
      ComponentProps<typeof DataView.Root<DataType>>,
      'dataMode' | 'state' | 'makeColumns'
    >;
    DataViewTableContent?: ComponentProps<typeof DataView.TableContent>;
    DataViewPagination?: ComponentProps<typeof DataView.Pagination>;
    DataViewSearchBar?: ComponentProps<typeof DataView.SearchBar>;
  };
}

export type { StudioDataViewToolbarProps } from '@nemo/common/src/components/DataView/StudioDataViewToolbar';
export { StudioDataViewToolbar } from '@nemo/common/src/components/DataView/StudioDataViewToolbar';

export const StudioDataView = <DataType,>({
  attributes,
  children,
  makeColumns,
  dataViewState,
  onRowClick,
  maxTwoLines = true,
  renderBulkActions,
  scrollContainerRef,
  searchField,
  toolbarSlotStart,
  toolbarSlotEnd,
}: Props<DataType>) => {
  const [showFilters, setShowFilters] = useState(false);
  const toggleFilters = useMemo(() => () => setShowFilters((prev) => !prev), []);
  const data = attributes?.DataViewRoot?.data ?? [];
  const totalCount = attributes?.DataViewRoot?.totalCount ?? data.length;
  const isEmpty = totalCount === 0;
  const {
    wrapColumns,
    onClick: rowClickHandler,
    onAuxClick: rowAuxClickHandler,
    className: rowClickClassName,
  } = useRowClick(onRowClick, data);

  const effectiveMakeColumns: typeof makeColumns = useMemo(() => {
    const withRowClick = wrapColumns(makeColumns);
    if (!maxTwoLines) return withRowClick;

    return (helper, prebuilt) => {
      const columns = withRowClick(helper, prebuilt);

      return columns.map((col) => {
        if (PREBUILT_COLUMN_IDS.has(col.id ?? '')) return col;

        const originalCell = col.cell;
        return {
          ...col,
          cell: (context: DataView.TanstackTable.CellContext<DataType, unknown>) => {
            const content =
              typeof originalCell === 'function'
                ? originalCell(context)
                : typeof originalCell === 'string'
                  ? originalCell
                  : context.renderValue();

            return (
              <div className="line-clamp-[2] [&_span]:whitespace-normal" data-testid="line-clamp">
                {content}
              </div>
            );
          },
        };
      });
    };
  }, [makeColumns, wrapColumns, maxTwoLines]);

  return (
    <DataView.Root
      dataMode="manual"
      state={dataViewState}
      data={data}
      makeColumns={effectiveMakeColumns}
      loadingRows={dataViewState.pagination.state.pageSize}
      {...attributes?.DataViewRoot}
      className={`studio-data-view-root ${attributes?.DataViewRoot?.className ?? ''}`}
    >
      <Stack className="relative flex-1 min-h-0 min-w-0 overflow-y-hidden" gap="density-xl">
        <StudioDataViewToolbar<DataType>
          searchField={searchField}
          showFilters={showFilters}
          onToggleFilters={toggleFilters}
          renderBulkActions={renderBulkActions}
          searchBarProps={attributes?.DataViewSearchBar}
          slotStart={toolbarSlotStart}
          slotEnd={toolbarSlotEnd}
        />
        <Flex className="min-h-0 h-full">
          {children ? (
            <Block ref={scrollContainerRef} className="flex-1 min-w-0 min-h-[300px] overflow-auto">
              {children}
            </Block>
          ) : (
            <div className="flex flex-col flex-1 min-w-0 min-h-0 bg-surface-raised border border-base rounded-lg overflow-hidden">
              <DataView.TableContent
                stickyTableHeader={attributes?.DataViewTableContent?.stickyTableHeader ?? true}
                onClick={rowClickHandler}
                onAuxClick={rowAuxClickHandler}
                renderEmptyState={() => (
                  <Block className="h-full">
                    <TableEmptyState
                      className="py-4"
                      header="No Entries Found"
                      emptyMessage="No entries available."
                    />
                  </Block>
                )}
                {...attributes?.DataViewTableContent}
                className={`studio-data-view-table flex-1 min-w-0 ${rowClickClassName} ${attributes?.DataViewTableContent?.className ?? ''}`}
              />
              <DataView.Pagination
                className="bg-surface-raised px-density-2xl py-density-lg"
                showItemsPerPage
                showWhileEmpty
                showWhileLessThanPageSize
                pageSizeOptions={DEFAULT_PAGE_SIZE_OPTIONS}
                {...attributes?.DataViewPagination}
              >
                <>
                  <PaginationControlsGroup>
                    <Text className="@max-2xl:hidden">Items per page</Text>
                    <PaginationPageSizeSelect />
                    {!isEmpty && (
                      <>
                        <PaginationDivider className="@max-lg:hidden" />
                        <PaginationItemRangeText className="@max-lg:hidden" />
                      </>
                    )}
                  </PaginationControlsGroup>
                  <PaginationNavigationGroup className="gap-2">
                    <PaginationArrowButton direction="first" />
                    <PaginationArrowButton direction="previous" />
                    <PaginationPageInput />
                    <PaginationPageCountText
                      pageCountTextFormatFn={(pageMeta) => `of ${pageMeta.total}`}
                    />
                    <PaginationArrowButton direction="next" />
                    <PaginationArrowButton direction="last" />
                  </PaginationNavigationGroup>
                </>
              </DataView.Pagination>
            </div>
          )}
          <FilterPanel
            showFilters={showFilters}
            containerTestId="studio-dataview-filter-panel-container"
            panelTestId="studio-dataview-filter-panel"
          >
            <ColumnFilterPanel />
          </FilterPanel>
        </Flex>
      </Stack>
    </DataView.Root>
  );
};
