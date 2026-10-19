// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@nvidia/foundations-react-core', () => ({
  Block: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Stack: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Text: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
  Flex: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PaginationArrowButton: () => null,
  PaginationDivider: () => null,
  PaginationControlsGroup: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PaginationItemRangeText: () => null,
  PaginationNavigationGroup: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PaginationPageCountText: () => null,
  PaginationPageInput: () => null,
  PaginationPageSizeSelect: () => null,
}));

vi.mock('@nemo/common/src/components/DataView/ColumnFilterPanel', () => ({
  ColumnFilterPanel: () => null,
}));

vi.mock('@nemo/common/src/components/DataView/FilterPanelToggle', () => ({
  FilterPanelToggle: () => null,
}));

vi.mock('@nemo/common/src/components/DataView/StudioAppliedFilters', () => ({
  StudioAppliedFilters: () => null,
}));

vi.mock('@nemo/common/src/components/DataView/FilterPanel', () => ({
  FilterPanel: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

// Shared state between Root and TableContent mocks so that wrapped makeColumns
// output (keyboard targets, line-clamp divs) is reflected in the rendered table.
let mockRenderedRows: React.ReactNode[][] = [];
let mockRootData: TestData[] = [];
interface FlatRow {
  item: TestData;
  index: number;
  depth: number;
  parentIndex?: number;
}
let mockFlatRows: FlatRow[] = [];

function flattenRows(data: TestData[]): FlatRow[] {
  const rows: FlatRow[] = [];
  data.forEach((item, i) => {
    rows.push({ item, index: i, depth: 0 });
    item.subRows?.forEach((sub, j) => {
      rows.push({ item: sub, index: j, depth: 1, parentIndex: i });
    });
  });
  return rows;
}

function renderCellContent(col: Record<string, unknown>, flatRow: FlatRow): React.ReactNode {
  if (col.cell && typeof col.cell === 'function') {
    return (col.cell as (ctx: Record<string, unknown>) => React.ReactNode)({
      row: {
        original: flatRow.item,
        index: flatRow.index,
        depth: flatRow.depth,
        getParentRow: () =>
          flatRow.parentIndex !== undefined
            ? { index: flatRow.parentIndex, original: mockRootData[flatRow.parentIndex] }
            : undefined,
      },
      getValue: () => flatRow.item[(col.accessorKey as keyof TestData) ?? 'name'],
      renderValue: () => String(flatRow.item[(col.accessorKey as keyof TestData) ?? 'name'] ?? ''),
    });
  }
  if (col.accessorKey) return String(flatRow.item[col.accessorKey as keyof TestData] ?? '');
  return null;
}

vi.mock('@nemo/common/src/components/DataView/internal', () => ({
  useInnerDataViewContext: () => ({
    table: { getAllLeafColumns: () => [], getSelectedRowModel: () => ({ flatRows: [] }) },
  }),
  Toolbar: ({
    children,
    slotBulkActions,
  }: React.PropsWithChildren<{ slotBulkActions?: React.ReactNode }>) => (
    <div>
      {slotBulkActions}
      {children}
    </div>
  ),
  SearchBar: () => <div data-testid="search-bar" />,
  BulkActions: ({
    children,
  }: {
    children: (ctx: { selectedRows: { original: unknown }[]; table: unknown }) => React.ReactNode;
  }) => (
    <div data-testid="bulk-actions">
      {children({
        selectedRows: mockRootData.map((item) => ({ original: item })),
        table: {},
      })}
    </div>
  ),
  Root: ({
    children,
    makeColumns,
    data,
  }: {
    children: React.ReactNode;
    makeColumns: (
      helper: {
        accessor: (field: string, opts?: Record<string, unknown>) => Record<string, unknown>;
      },
      prebuilt: Record<string, (opts?: Record<string, unknown>) => Record<string, unknown>>
    ) => Array<Record<string, unknown>>;
    data: TestData[];
  }) => {
    mockRootData = data ?? [];

    const columns =
      makeColumns?.(
        {
          accessor: (field: string, opts: Record<string, unknown> = {}) => ({
            id: field,
            accessorKey: field,
            ...opts,
          }),
        },
        {
          rowSelectionColumn: (opts: Record<string, unknown> = {}) => ({
            id: 'row-selection',
            ...opts,
          }),
          rowActionsColumn: (opts: Record<string, unknown> = {}) => ({
            id: 'row-actions',
            ...opts,
          }),
          rowExpansionColumn: (opts: Record<string, unknown> = {}) => ({
            id: 'row-expansion',
            ...opts,
          }),
        }
      ) ?? [];

    mockFlatRows = flattenRows(mockRootData);
    mockRenderedRows = mockFlatRows.map((flatRow) =>
      columns.map((col) => renderCellContent(col, flatRow))
    );

    return <div data-testid="dv-root">{children}</div>;
  },
  TableContent: ({
    className,
    onClick,
    onAuxClick,
  }: {
    className?: string;
    onClick?: React.MouseEventHandler;
    onAuxClick?: React.MouseEventHandler;
  }) => (
    // ref (not JSX onClick) so jsx-a11y doesn't flag the test-only <table>
    <table
      className={className}
      ref={(el) => {
        if (el) {
          el.onclick = onClick ? (onClick as unknown as (e: MouseEvent) => void) : null;
          el.onauxclick = onAuxClick ? (onAuxClick as unknown as (e: MouseEvent) => void) : null;
        }
      }}
    >
      <tbody>
        {mockRenderedRows.map((cells, rowIdx) => (
          <tr key={rowIdx} data-index={rowIdx}>
            {cells.map((cell, cellIdx) => (
              <td key={cellIdx}>{cell}</td>
            ))}
            <td>
              <button type="button">Delete {mockFlatRows[rowIdx]?.item.name}</button>
            </td>
            <td>
              <a href="/details">View {mockFlatRows[rowIdx]?.item.name}</a>
            </td>
            <td>
              <input type="text" defaultValue="edit" aria-label={`edit-${rowIdx}`} />
            </td>
            <td>
              <span data-no-row-click>Excluded {mockFlatRows[rowIdx]?.item.name}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  ),
  Pagination: ({ children }: React.PropsWithChildren) => (
    <div data-testid="dv-pagination">{children}</div>
  ),
  TanstackTable: {},
}));

type TestData = { id: string; name: string; subRows?: TestData[] };

function middleClick(element: Element) {
  const event = new MouseEvent('auxclick', { bubbles: true, cancelable: true, button: 1 });
  fireEvent(element, event);
  return event;
}

const testData: TestData[] = [
  { id: '1', name: 'Alice' },
  { id: '2', name: 'Bob' },
  { id: '3', name: 'Charlie' },
];

const mockDataViewState = {
  pagination: { state: { pageIndex: 0, pageSize: 50 }, set: vi.fn() },
  sorting: { state: [], set: vi.fn() },
  rowSelection: { state: {}, set: vi.fn() },
  columnPinning: { state: {}, set: vi.fn() },
  columnOrder: { state: [], set: vi.fn() },
  columnVisibility: { state: {}, set: vi.fn() },
  columnFiltering: { state: [], set: vi.fn() },
  searchBar: { state: '', set: vi.fn() },
  expansion: { state: {}, set: vi.fn() },
  rowHighlight: { state: undefined, set: vi.fn() },
  displayMode: { state: 'table' as const, set: vi.fn() },
  tab: { state: undefined, set: vi.fn() },
};

const baseMakeColumns = ({
  accessor,
}: {
  accessor: (field: string, opts?: Record<string, unknown>) => Record<string, unknown>;
}) => [accessor('name', { header: 'Name' })];

const defaultProps = {
  dataViewState: mockDataViewState as never,
  makeColumns: baseMakeColumns as never,
  attributes: {
    DataViewRoot: { data: testData },
  },
};

describe('StudioDataView', () => {
  beforeEach(() => {
    mockRenderedRows = [];
    mockRootData = [];
    mockFlatRows = [];
  });

  describe('search bar', () => {
    it('should render search bar when searchField is provided', () => {
      render(<StudioDataView {...defaultProps} searchField="name" />);
      expect(screen.getByTestId('search-bar')).toBeInTheDocument();
    });

    it('should not render search bar when searchField is omitted', () => {
      render(<StudioDataView {...defaultProps} />);
      expect(screen.queryByTestId('search-bar')).not.toBeInTheDocument();
    });
  });

  describe('rendering', () => {
    it('should not add cursor-pointer when onRowClick is not provided', () => {
      render(<StudioDataView {...defaultProps} />);
      const table = screen.getByRole('table');
      expect(table.className).not.toContain('cursor-pointer');
    });

    it('should add cursor-pointer class when onRowClick is provided', () => {
      render(<StudioDataView {...defaultProps} onRowClick={vi.fn()} />);
      const table = screen.getByRole('table');
      expect(table.className).toContain('cursor-pointer');
    });
  });

  describe('onRowClick', () => {
    it('should call onRowClick with row data when a cell is clicked', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Alice'));

      expect(onRowClick).toHaveBeenCalledWith(testData[0], 0, expect.anything());
    });

    it('should call onRowClick with correct row for second row', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Bob'));

      expect(onRowClick).toHaveBeenCalledWith(testData[1], 1, expect.anything());
    });

    it('should NOT call onRowClick when a button is clicked', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByRole('button', { name: 'Delete Alice' }));

      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('should NOT call onRowClick when a link is clicked', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByRole('link', { name: 'View Bob' }));

      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('should NOT call onRowClick when an input is clicked', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByRole('textbox', { name: 'edit-0' }));

      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('should NOT call onRowClick when an element with data-no-row-click is clicked', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Excluded Charlie'));

      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('should not attach click handler when onRowClick is not provided', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} />);

      fireEvent.click(screen.getByText('Alice'));

      expect(onRowClick).not.toHaveBeenCalled();
    });
  });

  describe('keyboard navigation', () => {
    it('should render a keyboard target in each row when onRowClick is provided', () => {
      render(<StudioDataView {...defaultProps} onRowClick={vi.fn()} />);

      const targets = screen.getAllByRole('button', { name: 'Open row' });
      expect(targets).toHaveLength(testData.length);
    });

    it('should NOT render keyboard targets when onRowClick is not provided', () => {
      render(<StudioDataView {...defaultProps} />);

      const targets = screen.queryAllByRole('button', { name: 'Open row' });
      expect(targets).toHaveLength(0);
    });

    it('should call onRowClick when Enter is pressed on a keyboard target', async () => {
      const user = userEvent.setup();
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      const targets = screen.getAllByRole('button', { name: 'Open row' });
      targets[0].focus();
      await user.keyboard('{Enter}');

      expect(onRowClick).toHaveBeenCalledWith(testData[0], 0, expect.anything());
    });

    it('should call onRowClick when Space is pressed on a keyboard target', async () => {
      const user = userEvent.setup();
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      const targets = screen.getAllByRole('button', { name: 'Open row' });
      targets[1].focus();
      await user.keyboard(' ');

      expect(onRowClick).toHaveBeenCalledWith(testData[1], 1, expect.anything());
    });

    it('should NOT call onRowClick for non-activation keys', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      const target = screen.getAllByRole('button', { name: 'Open row' })[0];
      fireEvent.keyDown(target, { key: 'Tab' });
      fireEvent.keyDown(target, { key: 'Escape' });
      fireEvent.keyDown(target, { key: 'a' });

      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('should inject keyboard target into first column even when it is a prebuilt column', () => {
      const makeColumnsWithPrebuilt = (
        {
          accessor,
        }: { accessor: (field: string, opts?: Record<string, unknown>) => Record<string, unknown> },
        {
          rowSelectionColumn,
        }: { rowSelectionColumn: (opts?: Record<string, unknown>) => Record<string, unknown> }
      ) => [rowSelectionColumn({ size: 48 }), accessor('name', { header: 'Name' })];

      render(
        <StudioDataView
          {...defaultProps}
          makeColumns={makeColumnsWithPrebuilt as never}
          onRowClick={vi.fn()}
        />
      );

      const targets = screen.getAllByRole('button', { name: 'Open row' });
      expect(targets).toHaveLength(testData.length);
    });
  });

  describe('maxTwoLines', () => {
    it('should wrap data cell content with line-clamp class by default', () => {
      render(<StudioDataView {...defaultProps} />);

      const clampedDivs = screen.getAllByTestId('line-clamp');
      expect(clampedDivs.length).toBeGreaterThan(0);
      expect(clampedDivs[0].className).toContain('line-clamp-');
    });

    it('should not wrap cell content when maxTwoLines is false', () => {
      render(<StudioDataView {...defaultProps} maxTwoLines={false} />);

      const clampedDivs = screen.queryAllByTestId('line-clamp');
      expect(clampedDivs).toHaveLength(0);
    });

    it('should not apply line-clamp to prebuilt columns', () => {
      const makeColumnsWithPrebuilt = (
        {
          accessor,
        }: { accessor: (field: string, opts?: Record<string, unknown>) => Record<string, unknown> },
        {
          rowSelectionColumn,
        }: { rowSelectionColumn: (opts?: Record<string, unknown>) => Record<string, unknown> }
      ) => [rowSelectionColumn({ size: 48 }), accessor('name', { header: 'Name' })];

      render(<StudioDataView {...defaultProps} makeColumns={makeColumnsWithPrebuilt as never} />);

      // Only data columns (name) get line-clamp wrappers, not prebuilt (row-selection)
      const clampedDivs = screen.getAllByTestId('line-clamp');
      expect(clampedDivs).toHaveLength(testData.length);
      expect(clampedDivs[0].className).toContain('line-clamp-');
    });
  });

  describe('sub-row click resolution', () => {
    // Visual layout: Parent(0) → Child-1(1) → Child-2(2) → Sibling(3)
    const dataWithSubRows: TestData[] = [
      {
        id: 'p1',
        name: 'Parent',
        subRows: [
          { id: 'c1', name: 'Child-1' },
          { id: 'c2', name: 'Child-2' },
        ],
      },
      { id: 's1', name: 'Sibling' },
    ];

    const subRowProps = {
      ...defaultProps,
      attributes: { DataViewRoot: { data: dataWithSubRows } },
    };

    it('should resolve the correct top-level row when sub-rows precede it', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...subRowProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Sibling'));

      expect(onRowClick).toHaveBeenCalledWith(dataWithSubRows[1], 1, expect.anything());
    });

    it('should resolve a sub-row correctly on click', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...subRowProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Child-1'));

      // Index is the parent's top-level data position (0), not the visual row position
      expect(onRowClick).toHaveBeenCalledWith(dataWithSubRows[0].subRows![0], 0, expect.anything());
    });

    it('should resolve a sub-row correctly on keyboard activation', async () => {
      const user = userEvent.setup();
      const onRowClick = vi.fn();
      render(<StudioDataView {...subRowProps} onRowClick={onRowClick} />);

      const targets = screen.getAllByRole('button', { name: 'Open row' });
      targets[1].focus();
      await user.keyboard('{Enter}');

      // Same index semantic as click: parent's top-level data position
      expect(onRowClick).toHaveBeenCalledWith(dataWithSubRows[0].subRows![0], 0, expect.anything());
    });

    it('should resolve the second sub-row correctly on click', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...subRowProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Child-2'));

      expect(onRowClick).toHaveBeenCalledWith(dataWithSubRows[0].subRows![1], 0, expect.anything());
    });

    it('should forward the event for sub-rows too', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...subRowProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Child-2'), { metaKey: true });

      expect(onRowClick).toHaveBeenCalledWith(dataWithSubRows[0].subRows![1], 0, expect.anything());
      expect(onRowClick.mock.calls[0][2].metaKey).toBe(true);
    });
  });

  describe('event forwarding for new-tab navigation', () => {
    it('should forward a plain click with no modifiers', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Bob'));

      const event = onRowClick.mock.calls[0][2];
      expect(onRowClick).toHaveBeenCalledWith(testData[1], 1, expect.anything());
      expect(event.metaKey).toBe(false);
      expect(event.ctrlKey).toBe(false);
      expect(event.button).toBe(0);
    });

    it('should forward the meta key so callers can open a new tab', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Bob'), { metaKey: true });

      expect(onRowClick).toHaveBeenCalledWith(testData[1], 1, expect.anything());
      expect(onRowClick.mock.calls[0][2].metaKey).toBe(true);
    });

    it('should forward the ctrl key', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      fireEvent.click(screen.getByText('Bob'), { ctrlKey: true });

      expect(onRowClick.mock.calls[0][2].ctrlKey).toBe(true);
    });

    it('should forward a middle click via onAuxClick', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      middleClick(screen.getByText('Bob'));

      expect(onRowClick).toHaveBeenCalledWith(testData[1], 1, expect.anything());
      expect(onRowClick.mock.calls[0][2].button).toBe(1);
    });

    it('should not fire on a middle click of an opted-out element', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      const event = middleClick(screen.getByRole('button', { name: 'Delete Alice' }));

      expect(onRowClick).not.toHaveBeenCalled();
      expect(event.defaultPrevented).toBe(false);
    });

    it('should suppress the row default on a middle click of a plain cell', () => {
      const onRowClick = vi.fn();
      render(<StudioDataView {...defaultProps} onRowClick={onRowClick} />);

      const event = middleClick(screen.getByText('Bob'));

      expect(onRowClick).toHaveBeenCalled();
      expect(event.defaultPrevented).toBe(true);
    });
  });

  describe('renderBulkActions', () => {
    it('should not render bulk actions when renderBulkActions is omitted', () => {
      render(<StudioDataView {...defaultProps} searchField="name" />);
      expect(screen.queryByTestId('bulk-actions')).not.toBeInTheDocument();
    });

    it('should render custom bulk actions when renderBulkActions is provided', () => {
      const renderBulkActions = vi.fn(({ selectedRows }: { selectedRows: TestData[] }) => (
        <button data-testid="custom-bulk-action">Delete ({selectedRows.length})</button>
      ));

      render(
        <StudioDataView
          {...defaultProps}
          searchField="name"
          renderBulkActions={renderBulkActions}
        />
      );

      expect(screen.getByTestId('custom-bulk-action')).toBeInTheDocument();
      expect(screen.getByTestId('custom-bulk-action')).toHaveTextContent(
        `Delete (${testData.length})`
      );
    });

    it('should pass unwrapped row data to renderBulkActions', () => {
      const renderBulkActions = vi.fn(() => <span>actions</span>);

      render(
        <StudioDataView
          {...defaultProps}
          searchField="name"
          renderBulkActions={renderBulkActions}
        />
      );

      expect(renderBulkActions).toHaveBeenCalledWith(
        expect.objectContaining({
          selectedRows: testData,
        })
      );
    });
  });
});
