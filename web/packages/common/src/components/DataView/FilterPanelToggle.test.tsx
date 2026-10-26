// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilterPanelToggle } from '@nemo/common/src/components/DataView/FilterPanelToggle';
import { render, screen, fireEvent } from '@testing-library/react';

const mockTable = {
  getAllLeafColumns: vi.fn(),
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
}));

describe('FilterPanelToggle', () => {
  it('returns null when no columns have getCanFilter', () => {
    mockTable.getAllLeafColumns.mockReturnValue([{ getCanFilter: () => false }]);

    const { container } = render(<FilterPanelToggle showFilters={false} onToggle={vi.fn()} />);

    expect(container.innerHTML).toBe('');
  });

  it('renders button when at least one column can filter', () => {
    mockTable.getAllLeafColumns.mockReturnValue([{ getCanFilter: () => true }]);

    render(<FilterPanelToggle showFilters={false} onToggle={vi.fn()} />);

    expect(screen.getByTestId('open-filters-button')).toBeInTheDocument();
    expect(screen.getByText('Filter')).toBeInTheDocument();
  });

  it('calls onToggle when button is clicked', () => {
    mockTable.getAllLeafColumns.mockReturnValue([{ getCanFilter: () => true }]);

    const onToggle = vi.fn();
    render(<FilterPanelToggle showFilters={false} onToggle={onToggle} />);

    fireEvent.click(screen.getByTestId('open-filters-button'));

    expect(onToggle).toHaveBeenCalledOnce();
  });

  it('sets aria-pressed=true when showFilters is true', () => {
    mockTable.getAllLeafColumns.mockReturnValue([{ getCanFilter: () => true }]);

    render(<FilterPanelToggle showFilters onToggle={vi.fn()} />);

    expect(screen.getByTestId('open-filters-button')).toHaveAttribute('aria-pressed', 'true');
  });

  it('sets aria-pressed=false when showFilters is false', () => {
    mockTable.getAllLeafColumns.mockReturnValue([{ getCanFilter: () => true }]);

    render(<FilterPanelToggle showFilters={false} onToggle={vi.fn()} />);

    expect(screen.getByTestId('open-filters-button')).toHaveAttribute('aria-pressed', 'false');
  });
});
