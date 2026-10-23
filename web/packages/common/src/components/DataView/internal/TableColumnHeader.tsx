// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DraggableAttributes, DraggableSyntheticListeners } from '@dnd-kit/core';
import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { useHandleResize } from '@nemo/common/src/components/DataView/internal/hooks/useResizableColumns';
import type { IntentionalAny } from '@nemo/common/src/components/DataView/internal/types';
import { getHeaderId } from '@nemo/common/src/components/DataView/internal/utils/header-utils';
import { ArrowUp, ArrowUpDown, GripEdgeVertical } from '@nemo/icons/icons';
import { Button, TableHeaderCell } from '@nvidia/foundations-react-core';
import { childrenToText } from '@nvidia/foundations-react-core/lib';
import { flexRender, type Header, type SortDirection } from '@tanstack/react-table';
import classnames from 'classnames';
import type { ComponentProps, JSX, ReactNode, Ref } from 'react';

interface DragProps {
  setNodeRef?: (node: HTMLElement | null) => void;
  setActivatorNodeRef?: (node: HTMLElement | null) => void;
  attributes: DraggableAttributes;
  listeners: DraggableSyntheticListeners;
  isDragging: boolean;
}

interface TableColumnHeaderProps extends ComponentProps<typeof TableHeaderCell> {
  automaticTitles?: boolean;
  header: Header<IntentionalAny, unknown>;
  dragProps?: DragProps;
}

export function TableColumnHeader({
  automaticTitles,
  className,
  dragProps,
  header,
  ...props
}: TableColumnHeaderProps): JSX.Element {
  const { isDataViewLoadingState, isDataViewErrorState } = useInnerDataViewContext();
  const children = header.isPlaceholder
    ? null
    : flexRender(header.column.columnDef.header, header.getContext());
  const headerMeta = header.column.columnDef.meta;
  return (
    <TableHeaderCell
      ref={dragProps?.setNodeRef as Ref<HTMLTableCellElement> | undefined}
      align={headerMeta?.headerAlignment ?? headerMeta?.alignment}
      className={classnames(
        className,
        'group',
        'data-[pinned]:sticky data-[pinned]:z-10',
        '[&>.data-view-header-control]:-mx-[var(--table-cell-inline-padding)] [&>.data-view-header-control]:-my-[var(--table-cell-block-padding)]',
        '[&>.data-view-header-control]:max-w-[calc(100%+var(--table-cell-inline-padding)*2-4px)]',
        dragProps?.isDragging && 'opacity-50'
      )}
      colSpan={header.column.columns.length > 1 ? header.column.columns.length : undefined}
      data-pinned={header.column.getIsPinned() || undefined}
      id={getHeaderId(header.id)}
      title={automaticTitles ? childrenToText(children as ReactNode) || undefined : undefined}
      {...props}
    >
      <TableHeaderControlCell
        disabled={isDataViewLoadingState || isDataViewErrorState}
        dragProps={dragProps}
        header={header}
      >
        {children as ReactNode}
      </TableHeaderControlCell>
      {!(isDataViewLoadingState || isDataViewErrorState) && header.column.getCanResize() && (
        <ColumnResizerHandle
          className="absolute top-0 right-0 z-10 group-hover:opacity-100"
          header={header}
        />
      )}
    </TableHeaderCell>
  );
}

interface TableHeaderControlCellProps {
  children: ReactNode;
  disabled: boolean;
  dragProps?: DragProps;
  header: Header<IntentionalAny, unknown>;
}

function TableHeaderControlCell({
  children,
  disabled,
  dragProps,
  header,
}: TableHeaderControlCellProps): JSX.Element {
  if (disabled) return <>{children}</>;

  const canSort = header.column.getCanSort();
  const sort = header.column.getIsSorted();
  const grip = dragProps?.listeners ? (
    <button
      ref={dragProps.setActivatorNodeRef}
      className="cursor-grab active:cursor-grabbing p-0.5 text-secondary hover:text-primary focus:outline-none shrink-0"
      aria-label="Drag to reorder column"
      type="button"
      {...dragProps.attributes}
      {...dragProps.listeners}
    >
      <GripEdgeVertical size="14" />
    </button>
  ) : null;

  // Neither interactive — plain label
  if (!canSort && !grip) return <>{children}</>;

  // Drag only — grip alongside plain label, no sort button
  if (!canSort) {
    return (
      <>
        {grip}
        {children}
      </>
    );
  }

  // Sort + drag — flex wrapper keeps them together without ButtonGroup chrome
  if (grip) {
    return (
      <div className="data-view-header-control flex items-center">
        {grip}
        <Button kind="tertiary" onClick={header.column.getToggleSortingHandler()}>
          <span className="truncate leading-[normal]">{children}</span>
          <SortIcon sort={sort} />
        </Button>
      </div>
    );
  }

  // Sort only — current behavior
  return (
    <Button
      className="data-view-header-control"
      kind="tertiary"
      onClick={header.column.getToggleSortingHandler()}
    >
      <span className="truncate leading-[normal]">{children}</span>
      <SortIcon sort={sort} />
    </Button>
  );
}

function SortIcon({ sort }: { sort: false | SortDirection }): JSX.Element {
  if (sort === false) {
    return <ArrowUpDown data-sorting-icon variant="line" />;
  }
  return (
    <ArrowUp
      className="transition-transform data-[descending]:rotate-180"
      data-descending={sort === 'desc' || undefined}
      data-selected
      data-sorting-icon
      variant="line"
    />
  );
}

interface ColumnResizerHandleProps {
  className?: string;
  header: Header<IntentionalAny, unknown>;
}

function ColumnResizerHandle({ className, header }: ColumnResizerHandleProps): JSX.Element {
  const { handleResize, handleDoubleClick } = useHandleResize(header);
  return (
    <div
      role="presentation"
      className={classnames(
        className,
        'h-full w-px cursor-col-resize touch-none bg-[var(--border-color-base)] opacity-0 transition-opacity select-none hover:w-0.5 hover:opacity-100',
        header.column.getIsResizing() && 'w-0.5 opacity-100'
      )}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={handleDoubleClick}
      onMouseDown={handleResize}
      onTouchStart={handleResize}
    />
  );
}
