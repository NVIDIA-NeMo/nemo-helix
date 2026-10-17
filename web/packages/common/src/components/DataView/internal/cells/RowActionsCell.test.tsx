// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  RowActionsCell,
  type RowActionsCellProps,
} from '@nemo/common/src/components/DataView/internal/cells/RowActionsCell';
import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal/context';
import { Root } from '@nemo/common/src/components/DataView/internal/Root';
import { useDataViewState } from '@nemo/common/src/components/DataView/internal/useDataViewState';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

interface TestRow {
  name: string;
}

type RowActions = RowActionsCellProps<TestRow>['rowActions'];

function Cell({ rowActions }: { rowActions?: RowActions }) {
  const { table } = useInnerDataViewContext();
  const ctx = table.getRowModel().rows[0].getAllCells()[0].getContext();
  return <RowActionsCell ctx={ctx} rowActions={rowActions} />;
}

function TestCell({ rowActions }: { rowActions?: RowActions }) {
  const state = useDataViewState();
  return (
    <Root
      data={[{ name: 'Insight' }]}
      state={state}
      makeColumns={({ accessor }) => [accessor('name', { header: 'Name' })]}
    >
      <Cell rowActions={rowActions} />
    </Root>
  );
}

describe('RowActionsCell', () => {
  it.each<{ items: ReturnType<NonNullable<RowActions>> }>([
    { items: undefined },
    { items: null },
    { items: false },
    { items: [] },
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

    await user.click(screen.getByRole('button', { name: 'Row Actions' }));
    await user.click(await screen.findByRole('menuitem', { name: 'View insight' }));

    expect(onSelect).toHaveBeenCalledOnce();
  });
});
