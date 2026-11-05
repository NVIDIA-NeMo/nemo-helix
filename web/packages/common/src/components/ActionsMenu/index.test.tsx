// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ActionsMenu,
  type ActionMenuItem,
  type DeleteAction,
} from '@nemo/common/src/components/ActionsMenu';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const openMenu = async (actions: ActionMenuItem[], deleteAction?: DeleteAction) => {
  const user = userEvent.setup();
  render(<ActionsMenu actions={actions} deleteAction={deleteAction} />);
  await user.click(screen.getByTestId('actions-menu-trigger'));
  return user;
};

describe('ActionsMenu', () => {
  it('renders the delete action last, in red, below a divider', async () => {
    const onDelete = vi.fn();
    const user = await openMenu([{ label: 'Edit', onSelect: vi.fn() }], { onSelect: onDelete });

    const items = screen.getAllByRole('menuitem');
    const separator = screen.getByRole('separator');
    expect(items.map((item) => item.textContent)).toEqual(['Edit', 'Delete']);
    expect(items[1]).toHaveClass('nv-menu-item--danger');
    expect(items[0].compareDocumentPosition(separator)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(separator.compareDocumentPosition(items[1])).toBe(Node.DOCUMENT_POSITION_FOLLOWING);

    await user.click(items[1]);
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it('adds no divider when the delete action is the only item', async () => {
    await openMenu([], { onSelect: vi.fn() });

    expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
    expect(screen.queryByRole('separator')).not.toBeInTheDocument();
  });

  it('reuses the divider of the last action instead of adding a second one', async () => {
    await openMenu([{ label: 'Rename', onSelect: vi.fn(), divider: {} }], { onSelect: vi.fn() });

    expect(screen.getAllByRole('separator')).toHaveLength(1);
  });

  it('ignores fields outside DeleteAction', async () => {
    const deleteAction = { onSelect: vi.fn(), description: 'Gone forever', divider: {} };
    await openMenu([], deleteAction);

    expect(screen.getByRole('menuitem')).toHaveTextContent(/^Delete$/);
    expect(screen.queryByRole('separator')).not.toBeInTheDocument();
  });

  it('uses the given label and disabled state for the delete action', async () => {
    await openMenu([{ label: 'Add to experiment', onSelect: vi.fn() }], {
      onSelect: vi.fn(),
      label: 'Remove from experiment',
      disabled: true,
    });

    expect(screen.getByRole('menuitem', { name: 'Remove from experiment' })).toBeDisabled();
  });
});
