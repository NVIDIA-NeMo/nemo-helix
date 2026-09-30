// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRowClick } from '@nemo/common/src/components/DataView/useRowClick';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FC } from 'react';

// Mock internal DataView module – useRowClick imports it only for TypeScript types
// but the module would still be loaded at runtime. Mocking avoids loading its full
// dependency tree (tanstack-table, KUI components, etc.) in this isolated test.
vi.mock('@nemo/common/src/components/DataView/internal', () => ({
  TanstackTable: {},
  Root: {},
}));

// ──────────────────────────────────────────────────────────────────────────
// Minimal test helpers
// ──────────────────────────────────────────────────────────────────────────

type Item = { id: string; name: string };

// Helper used by Harness to invoke the wrapped makeColumns function and render
// the resulting cell content for each row.
const HELPER = {
  accessor: (field: string, opts: Record<string, unknown> = {}) => ({
    id: field,
    accessorKey: field,
    ...opts,
  }),
};
const PREBUILT = {
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
};

// Minimal test harness that exercises useRowClick without the full DataView stack.
// It renders a <table> with one row per data item. Inside the first cell it places
// the keyboard target injected by wrapColumns, and the onClick handler is attached
// to the table element (same event-delegation pattern as the real TableContent).
interface HarnessProps {
  data: Item[];
  onRowClick?: (row: Item, index: number) => void;
}

const Harness: FC<HarnessProps> = ({ data, onRowClick }) => {
  const { wrapColumns, onClick, className } = useRowClick(onRowClick, data);

  // Build a single accessor column.
  const baseMakeColumns = (helper: typeof HELPER) => [
    helper.accessor('name', { header: 'Name' }),
  ];

  // wrapColumns(baseMakeColumns) returns a new MakeColumns function.
  // Invoke it immediately with the test helper/prebuilt to get the column array.
  const wrappedMakeColumns = wrapColumns(
    baseMakeColumns as Parameters<typeof wrapColumns>[0]
  );
  const columns = wrappedMakeColumns(
    HELPER as Parameters<typeof wrappedMakeColumns>[0],
    PREBUILT as Parameters<typeof wrappedMakeColumns>[1]
  );

  return (
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events,jsx-a11y/no-noninteractive-element-interactions
    <table className={className} onClick={onClick as React.MouseEventHandler<HTMLTableElement>}>
      <tbody>
        {data.map((item, index) => {
          const cellContent =
            typeof columns[0]?.cell === 'function'
              ? (columns[0].cell as (ctx: Record<string, unknown>) => React.ReactNode)({
                  row: {
                    original: item,
                    index,
                    depth: 0,
                    getParentRow: () => undefined,
                  },
                  getValue: () => item.name,
                  renderValue: () => item.name,
                })
              : item.name;

          return (
            <tr key={item.id} data-index={index}>
              <td>{cellContent}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
};

const testData: Item[] = [
  { id: '1', name: 'Alice' },
  { id: '2', name: 'Bob' },
];

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

describe('useRowClick – RowKeyboardTarget', () => {
  it('renders a native <button> element (not a span) for each row', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    expect(buttons).toHaveLength(testData.length);
    buttons.forEach((btn) => {
      expect(btn.tagName).toBe('BUTTON');
    });
  });

  it('sets type="button" on each keyboard target to prevent form submission', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons.forEach((btn) => {
      expect(btn).toHaveAttribute('type', 'button');
    });
  });

  it('labels each keyboard target as "Open row"', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    expect(screen.getAllByRole('button', { name: 'Open row' })).toHaveLength(testData.length);
  });

  it('applies the sr-only class for visual hiding', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons.forEach((btn) => {
      expect(btn).toHaveClass('sr-only');
    });
  });

  it('sets data-row-index to the correct row position', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    expect(buttons[0]).toHaveAttribute('data-row-index', '0');
    expect(buttons[1]).toHaveAttribute('data-row-index', '1');
  });

  it('does NOT set data-sub-index for top-level rows', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons.forEach((btn) => {
      expect(btn).not.toHaveAttribute('data-sub-index');
    });
  });

  it('calls onRowClick when the keyboard target button is clicked', () => {
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    fireEvent.click(buttons[0]);

    expect(onRowClick).toHaveBeenCalledWith(testData[0], 0);
  });

  it('calls onRowClick for the second row button click', () => {
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    fireEvent.click(buttons[1]);

    expect(onRowClick).toHaveBeenCalledWith(testData[1], 1);
  });

  it('activates via Enter key (native button behaviour)', async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons[0].focus();
    await user.keyboard('{Enter}');

    expect(onRowClick).toHaveBeenCalledWith(testData[0], 0);
  });

  it('activates via Space key (native button behaviour)', async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons[1].focus();
    await user.keyboard(' ');

    expect(onRowClick).toHaveBeenCalledWith(testData[1], 1);
  });

  it('does not render keyboard targets when onRowClick is undefined', () => {
    render(<Harness data={testData} />);

    expect(screen.queryAllByRole('button', { name: 'Open row' })).toHaveLength(0);
  });

  it('adds cursor-pointer class to the table when onRowClick is provided', () => {
    render(<Harness data={testData} onRowClick={vi.fn()} />);

    const table = screen.getByRole('table');
    expect(table.className).toContain('cursor-pointer');
  });

  it('does not add cursor-pointer class when onRowClick is undefined', () => {
    render(<Harness data={testData} />);

    const table = screen.getByRole('table');
    expect(table.className).not.toContain('cursor-pointer');
  });

  it('does not call onRowClick when Tab key is pressed (regression: old onKeyDown guard replaced by native button)', async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    const buttons = screen.getAllByRole('button', { name: 'Open row' });
    buttons[0].focus();
    await user.keyboard('{Tab}');

    expect(onRowClick).not.toHaveBeenCalled();
  });
});

describe('useRowClick – event delegation', () => {
  it('calls onRowClick when a non-interactive table cell is clicked', () => {
    const onRowClick = vi.fn();
    render(<Harness data={testData} onRowClick={onRowClick} />);

    fireEvent.click(screen.getByText('Alice'));

    expect(onRowClick).toHaveBeenCalledWith(testData[0], 0);
  });

  it('does not call onRowClick when no handler is attached', () => {
    const onRowClick = vi.fn();
    render(<Harness data={testData} />);

    fireEvent.click(screen.getByText('Alice'));

    expect(onRowClick).not.toHaveBeenCalled();
  });
});