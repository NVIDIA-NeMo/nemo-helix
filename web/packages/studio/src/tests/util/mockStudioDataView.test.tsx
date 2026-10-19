// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockStudioDataView } from '@studio/tests/util/mockStudioDataView';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

interface MenuItem {
  kind?: 'divider';
  children?: string;
  onSelect?: () => void;
}

type ActionItems = MenuItem[] | false | null | undefined;

function TestDataView({ items }: { items: ActionItems }) {
  return (
    <MockStudioDataView
      dataViewState={{}}
      attributes={{ DataViewRoot: { data: [{ name: 'Insight' }] } }}
      makeColumns={(_helper, prebuilt) => {
        const rowActionsColumn = prebuilt.rowActionsColumn as (options: {
          rowActions: () => ActionItems;
        }) => unknown;
        return [rowActionsColumn({ rowActions: () => items })];
      }}
    />
  );
}

describe('MockStudioDataView row actions', () => {
  it.each<{ items: ActionItems }>([
    { items: undefined },
    { items: null },
    { items: false },
    { items: [] },
    { items: [{ kind: 'divider' }] },
  ])('does not render an actions button for $items', ({ items }) => {
    render(<TestDataView items={items} />);

    expect(screen.queryByRole('button', { name: 'Actions' })).not.toBeInTheDocument();
  });

  it('opens and invokes a configured action alongside dividers', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<TestDataView items={[{ kind: 'divider' }, { children: 'View insight', onSelect }]} />);

    const trigger = screen.getByRole('button', { name: 'Actions' });
    await user.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    await user.click(screen.getByRole('menuitem', { name: 'View insight' }));

    expect(onSelect).toHaveBeenCalledOnce();
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
  });
});
