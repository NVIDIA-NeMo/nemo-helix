// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import { numberRangeFilter } from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter/util';
import { StudioAppliedFilters } from '@nemo/common/src/components/DataView/StudioAppliedFilters';
import { render, screen, fireEvent } from '@testing-library/react';

const mockSetColumnFilters = vi.fn();
const mockColumns = new Map<string, Record<string, unknown>>();

const mockTable = {
  getState: () => ({
    columnFilters: [...mockColumns.keys()].map((id) => ({ id, value: undefined })),
  }),
  getColumn: (id: string) => mockColumns.get(id),
  setColumnFilters: mockSetColumnFilters,
};

vi.mock('@nemo/common/src/components/DataView/internal', () => ({
  useInnerDataViewContext: () => ({ table: mockTable }),
}));

vi.mock('@nvidia/foundations-react-core', () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button {...props}>{children}</button>
  ),
  Flex: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tag: ({ children, onClick }: React.PropsWithChildren<{ onClick?: () => void }>) => (
    <button type="button" onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

function makeColumn(
  id: string,
  overrides: {
    filterValue?: unknown;
    meta?: Record<string, unknown>;
    header?: string;
  } = {}
) {
  const setFilterValue = vi.fn();
  const column = {
    id,
    columnDef: {
      header: overrides.header ?? id,
      meta: overrides.meta ?? { filter: { type: 'text' } },
    },
    getFilterValue: () => overrides.filterValue,
    setFilterValue,
  };
  return { column, setFilterValue };
}

describe('StudioAppliedFilters', () => {
  beforeEach(() => {
    mockColumns.clear();
    mockSetColumnFilters.mockClear();
  });

  it('returns null when no filters are applied', () => {
    const { container } = render(<StudioAppliedFilters />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a tag for a text filter with correct label and value', () => {
    const { column } = makeColumn('name', {
      filterValue: 'Alice',
      header: 'Name',
    });
    mockColumns.set('name', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Name:')).toBeInTheDocument();
    expect(screen.getByText('Alice')).toBeInTheDocument();
  });

  it('renders a tag for a single-select filter mapping value to option label', () => {
    const { column } = makeColumn('status', {
      filterValue: 'active',
      header: 'Status',
      meta: {
        filter: {
          type: 'single-select',
          options: [
            { value: 'active', label: 'Active' },
            { value: 'inactive', label: 'Inactive' },
          ],
        },
      },
    });
    mockColumns.set('status', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Status:')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
  });

  it('renders a tag for a multi-select filter (object/MultiState) mapping keys to option labels', () => {
    const { column } = makeColumn('kind', {
      filterValue: { lora: true, full: true },
      header: 'Kind',
      meta: {
        filter: {
          type: 'multi-select',
          options: [
            { value: 'lora', label: 'LoRA' },
            { value: 'full', label: 'Full' },
          ],
        },
      },
    });
    mockColumns.set('kind', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Kind:')).toBeInTheDocument();
    expect(screen.getByText('LoRA, Full')).toBeInTheDocument();
  });

  it('renders a tag for an array filter value showing element values not indices', () => {
    const { column } = makeColumn('base_model', {
      filterValue: ['llama-3', 'mistral-7b'],
      header: 'Base Model',
      meta: { filter: { type: 'custom', renderFilter: () => null } },
    });
    mockColumns.set('base_model', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Base Model:')).toBeInTheDocument();
    expect(screen.getByText('llama-3, mistral-7b')).toBeInTheDocument();
  });

  it('renders a tag for a datetime filter with formatted date range', () => {
    const { column } = makeColumn('created_at', {
      filterValue: {
        $gte: '2024-01-01T00:00:00.000Z',
        $lte: '2024-01-31T23:59:59.999Z',
      },
      header: 'Created At',
      meta: {
        filter: dateTimeFilter('Created At'),
      },
    });
    mockColumns.set('created_at', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Created At:')).toBeInTheDocument();
    const tag = screen.getByRole('button', { name: /Created At:/ });
    const [start, end] = (tag.textContent ?? '').split('—');
    expect(start).toMatch(/\b0?1\b/);
    expect(start).toContain('2024');
    expect(end).toMatch(/\b31\b/);
    expect(end).toContain('2024');
    expect(tag).not.toHaveTextContent('2023');
  });

  it('renders a tag for a numeric range filter with formatted bounds', () => {
    const { column } = makeColumn('duration', {
      filterValue: { $gte: 30, $lte: 480 },
      header: 'Duration',
      meta: { filter: numberRangeFilter('Duration (minutes)', { min: 0, max: 600, step: 15 }) },
    });
    mockColumns.set('duration', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Duration (minutes):')).toBeInTheDocument();
    expect(screen.getByText('30 – 480')).toBeInTheDocument();
  });

  it('renders a tag for an open-ended numeric range filter (lower bound only)', () => {
    const { column } = makeColumn('cost', {
      filterValue: { $gte: 10 },
      header: 'Cost',
      meta: { filter: numberRangeFilter('Cost (USD)', { min: 0, max: 50, step: 1 }) },
    });
    mockColumns.set('cost', column);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Cost (USD):')).toBeInTheDocument();
    expect(screen.getByText('≥ 10')).toBeInTheDocument();
  });

  it('renders Clear Filters button that calls table.setColumnFilters([])', () => {
    const { column } = makeColumn('name', { filterValue: 'test' });
    mockColumns.set('name', column);

    render(<StudioAppliedFilters />);

    fireEvent.click(screen.getByTestId('clear-filters'));

    expect(mockSetColumnFilters).toHaveBeenCalledWith([]);
  });

  it('clicking a filter tag calls column.setFilterValue(undefined)', () => {
    const { column, setFilterValue } = makeColumn('name', {
      filterValue: 'Alice',
      header: 'Name',
    });
    mockColumns.set('name', column);

    render(<StudioAppliedFilters />);

    fireEvent.click(screen.getByRole('button', { name: /Alice/ }));

    expect(setFilterValue).toHaveBeenCalledWith(undefined);
  });

  it('skips columns without meta.filter defined', () => {
    const { column: filterColumn } = makeColumn('name', {
      filterValue: 'Alice',
      header: 'Name',
    });
    const noFilterColumn = {
      id: 'id',
      columnDef: { header: 'ID', meta: {} },
      getFilterValue: () => '123',
      setFilterValue: vi.fn(),
    };
    mockColumns.set('name', filterColumn);
    mockColumns.set('id', noFilterColumn);

    render(<StudioAppliedFilters />);

    expect(screen.getByText('Name:')).toBeInTheDocument();
    expect(screen.queryByText('ID:')).not.toBeInTheDocument();
  });
});
