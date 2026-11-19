// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileActions } from '@studio/components/FilesetFilePreviewPanel/components/FileActions';
import type { FileSystemFile } from '@studio/components/FilesTable/utils';
import { render } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@studio/providers/workers/useWorkers', () => ({
  useWorkers: () => ({ createWorker: vi.fn() }),
}));

vi.mock('@studio/components/FilesTable/RenameFileModal', () => ({
  RenameFileModal: () => <div data-testid="rename-file-modal" />,
}));

vi.mock('@studio/components/FilesTable/CreateFileSplitsModal', () => ({
  CreateFileSplitsModal: () => <div data-testid="create-file-splits-modal" />,
}));

describe('FileActions', () => {
  const file: FileSystemFile = { type: 'file', path: 'folder/data.json', size: 100, oid: 'oid-1' };

  const openMenu = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole('button', { name: /open file actions menu/i }));
  };

  it('shows the actions with Delete last and styled as danger', async () => {
    const user = userEvent.setup();
    render(<FileActions workspace="default" filesetName="test-dataset" file={file} />);

    await openMenu(user);

    const items = screen.getAllByTestId('quick-actions-menu-item');
    expect(items.map((item) => item.textContent)).toEqual([
      'Create Split',
      'Download',
      'Rename',
      'Delete',
    ]);
    expect(items.at(-1)).toHaveClass('nv-menu-item--danger');
    expect(items[0]).not.toHaveClass('nv-menu-item--danger');
  });

  it('opens the delete confirmation modal when Delete is clicked', async () => {
    const user = userEvent.setup();
    render(<FileActions workspace="default" filesetName="test-dataset" file={file} />);

    await openMenu(user);
    await user.click(screen.getByText('Delete'));

    expect(
      await screen.findByRole('heading', { name: "Delete file 'folder/data.json'?" })
    ).toBeInTheDocument();
  });
});
