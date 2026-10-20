// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RowActionsCellProps } from '@nemo/common/src/components/DataView/internal/cells/RowActionsCell';
import { Root } from '@nemo/common/src/components/DataView/internal/Root';
import { TableContent } from '@nemo/common/src/components/DataView/internal/TableContent';
import { useDataViewState } from '@nemo/common/src/components/DataView/internal/useDataViewState';
import type { DropdownProps } from '@nvidia/foundations-react-core';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

interface TestRow {
  name: string;
}

type RowActions = RowActionsCellProps<TestRow>['rowActions'];

interface TestCellProps {
  rowActions?: RowActions;
  cellProps?: Partial<Omit<DropdownProps, 'items'>>;
}

function TestCell({ rowActions, cellProps }: TestCellProps) {
  const state = useDataViewState();
  return (
    <Root
      data={[{ name: 'Insight' }]}
      state={state}
      makeColumns={({ accessor }, { rowActionsColumn }) => [
        accessor('name', { header: 'Name' }),
        rowActionsColumn({ rowActions, cellProps }),
      ]}
    >
      <TableContent />
    </Root>
  );
}

describe('RowActionsCell', () => {
  it.each<{ items: ReturnType<NonNullable<RowActions>> }>([
    { items: undefined },
    { items: null },
    { items: false },
    { items: [] },
    { items: [{ kind: 'divider' }] },
    { items: [{ kind: 'divider' }, { kind: 'divider' }] },
  ])('does not render a menu when rowActions returns $items', ({ items }) => {
    render(<TestCell rowActions={() => items} />);

    expect(screen.queryByRole('button', { name: 'Row Actions' })).not.toBeInTheDocument();
  });

  it('does not render a menu when rowActions is omitted', () => {
    render(<TestCell />);

    expect(screen.queryByRole('button', { name: 'Row Actions' })).not.toBeInTheDocument();
  });

  it('renders and invokes a configured action', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<TestCell rowActions={() => [{ children: 'View insight', onSelect }]} />);

    const trigger = screen.getByRole('button', { name: 'Row Actions' });
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    await user.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    await user.click(await screen.findByRole('menuitem', { name: 'View insight' }));

    expect(onSelect).toHaveBeenCalledOnce();
  });

  it('keeps a menu with dividers and a configured action', () => {
    render(<TestCell rowActions={() => [{ kind: 'divider' }, { children: 'View insight' }]} />);

    expect(screen.getByRole('button', { name: 'Row Actions' })).toBeInTheDocument();
  });

  it('keeps string menu entries', () => {
    render(<TestCell rowActions={() => ['View insight']} />);

    expect(screen.getByRole('button', { name: 'Row Actions' })).toBeInTheDocument();
  });

  it('passes cellProps through rowActionsColumn', () => {
    render(
      <TestCell
        rowActions={() => [{ children: 'View insight' }]}
        cellProps={{ disabled: true, 'aria-label': 'Insight actions' }}
      />
    );

    expect(screen.getByRole('button', { name: 'Insight actions' })).toBeDisabled();
  });
});
