// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as DataView from '@nemo/common/src/components/DataView/internal';
import { ComponentProps, useCallback } from 'react';

const INTERACTIVE_ELEMENT_SELECTOR =
  'button, a, input, select, textarea, [role="button"], [data-no-row-click]';

const KEYBOARD_TARGET_SELECTOR = '[data-row-click]';

type MakeColumns<T> = ComponentProps<typeof DataView.Root<T>>['makeColumns'];

export type RowClickHandler<DataType> = (
  row: DataType,
  index: number,
  event: React.MouseEvent
) => void;

function findRowFromEvent(
  target: EventTarget | null
): { tr: HTMLTableRowElement; index: number } | null {
  const el = target as HTMLElement | null;
  if (!el) return null;
  const tr = el.closest('tr[data-index]') as HTMLTableRowElement | null;
  if (!tr) return null;
  const index = parseInt(tr.getAttribute('data-index')!, 10);
  if (isNaN(index)) return null;
  return { tr, index };
}

function isInteractiveElement(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return el.closest(INTERACTIVE_ELEMENT_SELECTOR) !== null;
}

/**
 * Resolve the correct data item and top-level data index for a clicked row.
 * Reads `data-row-index` (always the top-level data array position) and optionally
 * `data-sub-index` (position within parent's subRows) from the keyboard target.
 */
function resolveRowData<T>(
  data: T[],
  tr: HTMLTableRowElement
): { item: T; index: number } | undefined {
  const target = tr.querySelector<HTMLElement>(KEYBOARD_TARGET_SELECTOR);
  const rawIndex = target?.getAttribute('data-row-index');
  if (rawIndex == null) return undefined;

  const index = parseInt(rawIndex, 10);
  const parent = data[index];
  if (!parent) return undefined;

  const subIndex = target?.getAttribute('data-sub-index');
  if (subIndex != null) {
    const subRows = (parent as Record<string, unknown>).subRows as T[] | undefined;
    const item = subRows?.[parseInt(subIndex, 10)];
    return item ? { item, index } : undefined;
  }
  return { item: parent, index };
}

/**
 * Visually hidden, keyboard-focusable target rendered inside the first column
 * of each row. Enables keyboard navigation (Tab to focus, Enter/Space to activate)
 * without any post-render DOM manipulation.
 *
 * Also carries data attributes for sub-row resolution during event-delegated clicks.
 */
function RowKeyboardTarget({
  onActivate,
  dataIndex,
  subIndex,
}: {
  onActivate: (event: React.MouseEvent) => void;
  dataIndex: number;
  subIndex?: number;
}) {
  return (
    <button
      type="button"
      className="sr-only"
      aria-label="Open row"
      data-row-click
      data-row-index={dataIndex}
      {...(subIndex !== undefined && { 'data-sub-index': subIndex })}
      onClick={onActivate}
    />
  );
}

/**
 * Encapsulates row-click and keyboard-activation behaviour for StudioDataView.
 *
 * Returns three values the consumer applies:
 *  - `wrapColumns` — wraps a `makeColumns` function to inject a keyboard target
 *    into the first column of every row.
 *  - `onClick` — click handler for the table wrapper (event-delegation).
 *  - `className` — cursor-pointer styling applied when active.
 *
 * For tables with sub-rows, the keyboard target carries `data-row-index`
 * (top-level position in the data array) and `data-sub-index` (position
 * within parent's subRows) so the click handler can resolve
 * `data[index].subRows[subIndex]` correctly.
 *
 * When `onRowClick` is `undefined` all three are no-ops/empty, so the component
 * can be toggled with zero conditional logic.
 */
export function useRowClick<DataType>(
  onRowClick: RowClickHandler<DataType> | undefined,
  data: DataType[]
) {
  const activate = useCallback(
    (e: React.MouseEvent) => {
      if (!onRowClick) return;

      // If the user is selecting text, don't trigger the row click
      const selection = window.getSelection();
      if (selection && selection.toString().length > 0) return;

      if (isInteractiveElement(e.target)) return;
      const row = findRowFromEvent(e.target);
      if (!row) return;

      const resolved = resolveRowData(data, row.tr);
      if (resolved) {
        onRowClick(resolved.item, resolved.index, e);
      }
    },
    [onRowClick, data]
  );

  const onAuxClick: React.MouseEventHandler = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 1) return;
      if (isInteractiveElement(e.target)) return;
      e.preventDefault();
      activate(e);
    },
    [activate]
  );

  const wrapColumns = useCallback(
    (mc: MakeColumns<DataType>): MakeColumns<DataType> => {
      if (!onRowClick) return mc;

      return (helper, prebuilt) => {
        const columns = mc(helper, prebuilt);
        let injected = false;

        return columns.map((col) => {
          if (injected) return col;
          injected = true;

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

              const isSubRow = context.row.depth > 0;
              const parentRow = isSubRow ? context.row.getParentRow() : undefined;
              const topLevelIndex = parentRow?.index ?? context.row.index;

              return (
                <>
                  <RowKeyboardTarget
                    onActivate={(e) => onRowClick(context.row.original, topLevelIndex, e)}
                    dataIndex={topLevelIndex}
                    subIndex={isSubRow ? context.row.index : undefined}
                  />
                  {content}
                </>
              );
            },
          };
        });
      };
    },
    [onRowClick]
  );

  return {
    wrapColumns,
    onClick: onRowClick ? activate : undefined,
    onAuxClick: onRowClick ? onAuxClick : undefined,
    className: onRowClick ? '[&_tbody_tr]:cursor-pointer' : '',
  };
}
