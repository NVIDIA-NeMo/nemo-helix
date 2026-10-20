// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Mock factory for @nemo/common/src/components/DataView/StudioDataView.
 *
 * vi.mock is hoisted and statically analyzed per-file, so consumers must call
 * vi.mock in their own test file. This factory provides the shared implementation.
 *
 * Usage:
 *   vi.mock('@nemo/common/src/components/DataView/StudioDataView', async () => {
 *     const { studioDataViewMock } = await import('@studio/tests/util/mockStudioDataView');
 *     return studioDataViewMock();
 *   });
 *
 * The async dynamic-import pattern is required because vi.mock is hoisted before
 * ES imports execute, so a top-level import of studioDataViewMock would be in the
 * temporal dead zone when the factory runs.
 *
 * The mock renders:
 *   - A `data-testid="studio-data-view"` container with `aria-busy="true"` when loading
 *   - Column headers derived from makeColumns (via accessor key and header string)
 *   - One `data-testid="studio-data-view-row"` <tr> per flattened row (parent then each
 *     `subRows` entry, recursively), matching TanStack-style nested data
 *   - Cell values accessed by column id from each row object
 *   - When `rowActionsColumn({ rowActions })` is used, an Actions control opens a simple
 *     `role="menu"` with `role="menuitem"` entries (dividers skipped)
 */

import React, { useMemo, useState } from 'react';

type AnyObject = Record<string, unknown>;

const PREBUILT_IDS = new Set(['row-selection', 'row-actions', 'row-expansion']);

interface MockColumn {
  id: string;
  header: string;
}

interface BuildColumnsResult {
  columns: MockColumn[];
  rowActions?: MockRowActions;
}

function buildColumns(
  makeColumns: (helper: AnyObject, prebuilt: AnyObject) => unknown[]
): BuildColumnsResult {
  const captured: MockColumn[] = [];
  let rowActions: MockRowActions | undefined;

  const mockHelper: AnyObject = {
    accessor: (
      key: string | ((row: unknown) => unknown),
      options: { header?: unknown; id?: string } = {}
    ) => {
      const id = typeof key === 'string' ? key : (options.id ?? 'unknown');
      captured.push({ id, header: typeof options.header === 'string' ? options.header : id });
      return { id, ...options };
    },
    display: (options: { id?: string; header?: unknown } = {}) => {
      const id = options.id ?? 'display';
      if (!PREBUILT_IDS.has(id)) {
        captured.push({ id, header: typeof options.header === 'string' ? options.header : id });
      }
      return { id, ...options };
    },
  };

  const mockPrebuilt: AnyObject = {
    rowSelectionColumn: () => ({ id: 'row-selection' }),
    rowActionsColumn: (opts: { rowActions?: MockRowActions } = {}) => {
      if (typeof opts.rowActions === 'function') {
        rowActions = opts.rowActions;
      }
      captured.push({ id: 'row-actions', header: '' });
      return { id: 'row-actions', ...opts };
    },
    rowExpansionColumn: () => ({ id: 'row-expansion' }),
  };

  makeColumns(mockHelper, mockPrebuilt);
  return { columns: captured, rowActions };
}

interface RowMenuItem {
  kind?: 'divider';
  children?: React.ReactNode;
  onSelect?: () => void;
  disabled?: boolean;
}

type MockRowActions = (row: unknown) => RowMenuItem[] | false | null | undefined;

function MockRowActionsCell<DataType>({
  row,
  rowActions,
}: {
  row: DataType;
  rowActions: MockRowActions;
}) {
  const [open, setOpen] = useState(false);
  const items = rowActions(row);

  if (!items || !items.some((item) => item.kind !== 'divider')) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        aria-label="Actions"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        Actions
      </button>
      {open ? (
        <ul>
          {items.map((item, i) => {
            if (item.kind === 'divider') {
              return <li key={i} />;
            }
            return (
              <li key={i}>
                <button
                  type="button"
                  role="menuitem"
                  disabled={Boolean(item.disabled)}
                  onClick={(e) => {
                    e.stopPropagation();
                    item.onSelect?.();
                    setOpen(false);
                  }}
                >
                  {item.children}
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </>
  );
}

/** Depth-first flatten of `subRows` (same shape as StudioDataView / TanStack nested rows). */
function flattenWithSubRows<DataType>(rows: DataType[]): DataType[] {
  const out: DataType[] = [];
  for (const row of rows) {
    out.push(row);
    const subRows = (row as AnyObject).subRows as DataType[] | undefined;
    if (Array.isArray(subRows) && subRows.length > 0) {
      out.push(...flattenWithSubRows(subRows));
    }
  }
  return out;
}

interface MockProps<DataType> {
  makeColumns: (helper: AnyObject, prebuilt: AnyObject) => unknown[];
  dataViewState: unknown;
  onRowClick?: (row: DataType, index: number) => void;
  attributes?: {
    DataViewRoot?: {
      data?: DataType[];
      requestStatus?: string;
      [key: string]: unknown;
    };
    DataViewTableContent?: {
      renderEmptyState?: () => React.ReactNode;
      renderErrorState?: () => React.ReactNode;
      [key: string]: unknown;
    };
  };
  children?: React.ReactNode;
  [key: string]: unknown;
}

export const MockStudioDataView = <DataType,>({
  makeColumns,
  onRowClick,
  attributes,
  children,
}: MockProps<DataType>) => {
  const data = useMemo(() => {
    return (attributes?.DataViewRoot?.data ?? []) as DataType[];
  }, [attributes?.DataViewRoot?.data]);
  const { columns, rowActions } = useMemo(() => buildColumns(makeColumns), [makeColumns]);
  const displayRows = useMemo(() => flattenWithSubRows(data), [data]);

  const requestStatus = attributes?.DataViewRoot?.requestStatus;
  const isLoading = requestStatus === 'loading';
  const isError = requestStatus === 'error';
  const isEmpty = !isLoading && !isError && data.length === 0;
  return (
    <div data-testid="studio-data-view" aria-busy={isLoading ? true : undefined}>
      {isError && attributes?.DataViewTableContent?.renderErrorState?.()}
      {isEmpty && attributes?.DataViewTableContent?.renderEmptyState?.()}
      {!isLoading && !isError && !isEmpty && (
        <table>
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.id} scope="col">
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, index) => (
              <tr
                key={index}
                data-testid="studio-data-view-row"
                onClick={() => onRowClick?.(row, index)}
                // eslint-disable-next-line no-restricted-syntax
                style={onRowClick ? { cursor: 'pointer' } : undefined}
              >
                {columns.map((col) => {
                  if (col.id === 'row-actions' && rowActions) {
                    return (
                      <td key={col.id}>
                        <MockRowActionsCell row={row} rowActions={rowActions} />
                      </td>
                    );
                  }
                  const value = (row as AnyObject)[col.id];
                  return (
                    <td key={col.id}>
                      {value !== undefined && value !== null ? String(value) : ''}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {children}
    </div>
  );
};
