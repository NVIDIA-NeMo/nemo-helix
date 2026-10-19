// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ColumnFilterPanel } from '@nemo/common/src/components/DataView/ColumnFilterPanel';
import { FilterPanel } from '@nemo/common/src/components/DataView/FilterPanel';
import { FilterPanelToggle } from '@nemo/common/src/components/DataView/FilterPanelToggle';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import '@nemo/common/src/components/DataView/StudioDataView.css';
import { StudioAppliedFilters } from '@nemo/common/src/components/DataView/StudioAppliedFilters';
import { RowClickHandler, useRowClick } from '@nemo/common/src/components/DataView/useRowClick';
import { TableEmptyState } from '@nemo/common/src/components/TableEmptyState';
import { DEFAULT_PAGE_SIZE_OPTIONS } from '@nemo/common/src/constants/pagination';
import {
  Block,
  Flex,
  PaginationArrowButton,
  PaginationControlsGroup,
  PaginationDivider,
  PaginationItemRangeText,
  PaginationNavigationGroup,
  PaginationPageCountText,
  PaginationPageInput,
  PaginationPageSizeSelect,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  type ComponentProps,
  type ReactNode,
  type RefObject,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { createPortal } from 'react-dom';

const PREBUILT_COLUMN_IDS = new Set(['row-selection', 'row-actions', 'row-expansion']);

interface IntakeTelemetryToolbarProps<DataType = unknown> {
  searchField?: string;
  showFilters: boolean;
  onToggleFilters: () => void;
  slotEndPortalTargetId?: string;
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: DataView.TanstackTable.Table<DataType>;
  }) => ReactNode;
  searchBarProps?: ComponentProps<typeof DataView.SearchBar>;
  slotEnd?: ReactNode;
}

const IntakeTelemetryToolbar = <DataType,>({
  searchField,
  showFilters,
  onToggleFilters,
  slotEndPortalTargetId,
  renderBulkActions,
  searchBarProps,
  slotEnd,
}: IntakeTelemetryToolbarProps<DataType>) => {
  const { table } = DataView.useInnerDataViewContext();
  const hasFilterableColumns = table.getAllLeafColumns().some((col) => col.getCanFilter());
  const [portalTarget, setPortalTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalTarget(slotEndPortalTargetId ? document.getElementById(slotEndPortalTargetId) : null);
  }, [slotEndPortalTargetId]);

  const filterToggle = hasFilterableColumns ? (
    <FilterPanelToggle showFilters={showFilters} onToggle={onToggleFilters} />
  ) : null;
  const shouldPortalToggle = Boolean(slotEndPortalTargetId);
  const rendersInlineToggle = Boolean(!slotEndPortalTargetId && filterToggle);
  const rendersToolbar = Boolean(
    searchField || (!shouldPortalToggle && slotEnd) || rendersInlineToggle || renderBulkActions
  );

  return (
    <>
      {shouldPortalToggle && portalTarget
        ? createPortal(
            <>
              {filterToggle}
              {slotEnd}
            </>,
            portalTarget
          )
        : null}
      {rendersToolbar && (
        <DataView.Toolbar
          slotBulkActions={
            renderBulkActions ? (
              <DataView.BulkActions>
                {({ selectedRows, table }) =>
                  renderBulkActions({
                    selectedRows: selectedRows.map((row) => row.original) as DataType[],
                    table: table as DataView.TanstackTable.Table<DataType>,
                  })
                }
              </DataView.BulkActions>
            ) : undefined
          }
        >
          {searchField && (
            <DataView.SearchBar
              placeholder={searchBarProps?.placeholder ?? 'Search...'}
              {...searchBarProps}
            />
          )}
          {!shouldPortalToggle && slotEnd}
          {rendersInlineToggle ? filterToggle : null}
        </DataView.Toolbar>
      )}
      <StudioAppliedFilters />
    </>
  );
};

export interface IntakeTelemetryDataViewProps<DataType> {
  dataViewState: DataView.DataViewState;
  makeColumns: ComponentProps<typeof DataView.Root<DataType>>['makeColumns'];
  onRowClick?: RowClickHandler<DataType>;
  maxTwoLines?: boolean;
  searchField?: string;
  renderBulkActions?: (props: {
    selectedRows: DataType[];
    table: DataView.TanstackTable.Table<DataType>;
  }) => ReactNode;
  children?: ReactNode;
  toolbarSlotEnd?: ReactNode;
  slotEndPortalTargetId?: string;
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

export const IntakeTelemetryDataView = <DataType,>({
  attributes,
  children,
  makeColumns,
  dataViewState,
  onRowClick,
  maxTwoLines = true,
  renderBulkActions,
  scrollContainerRef,
  searchField,
  toolbarSlotEnd,
  slotEndPortalTargetId,
}: IntakeTelemetryDataViewProps<DataType>) => {
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
  }, [makeColumns, maxTwoLines, wrapColumns]);

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
        <IntakeTelemetryToolbar<DataType>
          searchField={searchField}
          showFilters={showFilters}
          onToggleFilters={toggleFilters}
          slotEndPortalTargetId={slotEndPortalTargetId}
          renderBulkActions={renderBulkActions}
          searchBarProps={attributes?.DataViewSearchBar}
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
                      header="No Data Found"
                      emptyMessage="No telemetry data available."
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
