// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FileQuickActions } from '@studio/components/FilesTable/FileQuickActions';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { TestProviders } from '@studio/tests/util/TestProviders';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('@studio/providers/workers/useWorkers', () => ({
  useWorkers: () => ({
    createWorker: vi.fn(),
  }),
}));

vi.mock('@studio/hooks/useSelectedDatasetId', () => ({
  useSelectedDatasetId: () => 'default/test',
}));

const mockFile: FileSystemNode = {
  path: 'test.json',
  size: 1024,
  type: 'file',
  oid: '123',
};

const mockTextFile: FileSystemNode = {
  path: 'test.txt',
  size: 1024,
  type: 'file',
  oid: '456',
};

const user = userEvent.setup();

const rootTestId = 'quick-actions-menu-trigger';

const openMenu = async () => {
  const triggers = await screen.findAllByTestId(rootTestId);
  expect(triggers).not.toHaveLength(0);
  await user.click(triggers[0]);
};

const renderComponent = (props = {}) => {
  return render(
    <TestProviders>
      <FileQuickActions datasetId="default/test" file={mockFile} {...props} />
    </TestProviders>
  );
};

describe('FileQuickActions', () => {
  it('renders limited actions for read-only dataset', async () => {
    renderComponent();
    await openMenu();
    expect(screen.getByText('Download')).toBeInTheDocument();
    expect(screen.getByText('Copy Path')).toBeInTheDocument();
    expect(screen.queryByText('Delete')).not.toBeInTheDocument();
    expect(screen.queryByText('Move')).not.toBeInTheDocument();
    expect(screen.queryByText('Rename')).not.toBeInTheDocument();
  });

  it('renders full actions for read/write dataset (local or s3)', async () => {
    renderComponent({ isReadWriteDataset: true });
    await openMenu();
    expect(screen.getByText('Download')).toBeInTheDocument();
    expect(screen.getByText('Copy Path')).toBeInTheDocument();
    expect(screen.getByText('Move')).toBeInTheDocument();
    expect(screen.getByText('Duplicate')).toBeInTheDocument();
    expect(screen.getByText('Create Split')).toBeInTheDocument();
    expect(screen.getByText('Transform')).toBeInTheDocument();
    expect(screen.getByText('Rename')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  it('uses currentFolder prop instead of query params', async () => {
    const onViewFile = vi.fn();
    renderComponent({
      currentFolder: 'training/data',
      onViewFile,
    });

    await openMenu();
    const viewButton = screen.getByText('View File');
    await user.click(viewButton);

    // Should call with full path including currentFolder
    expect(onViewFile).toHaveBeenCalledWith('training/data/test.json');
  });

  it('uses full file path for nested files without mangling by currentFolder', async () => {
    const onViewFile = vi.fn();
    const nestedFile: FileSystemNode = {
      path: 'training/data/nested.txt',
      size: 100,
      type: 'file',
      oid: 'oid-nested',
    };
    renderComponent({
      file: nestedFile,
      currentFolder: 'training',
      onViewFile,
    });

    await openMenu();
    await user.click(screen.getByText('View File'));

    expect(onViewFile).toHaveBeenCalledWith('training/data/nested.txt');
  });

  it('calls onViewFile callback when View File clicked', async () => {
    const onViewFile = vi.fn();
    renderComponent({ onViewFile });

    await openMenu();
    const viewButton = screen.getByText('View File');
    await user.click(viewButton);

    expect(onViewFile).toHaveBeenCalledWith('test.json');
  });

  it('hides View File when onViewFile not provided', async () => {
    // Without callback
    renderComponent();
    await openMenu();
    expect(screen.queryByText('View File')).not.toBeInTheDocument();
  });

  it('shows View File when onViewFile is provided', async () => {
    // With callback
    renderComponent({ onViewFile: vi.fn() });
    await openMenu();
    expect(screen.getByText('View File')).toBeInTheDocument();
  });

  it('shows View File for all file types when onViewFile provided', async () => {
    const onViewFile = vi.fn();
    renderComponent({ file: mockTextFile, onViewFile });

    await openMenu();
    expect(screen.getByText('View File')).toBeInTheDocument();
  });

  it('still shows other actions without onViewFile', async () => {
    renderComponent();

    await openMenu();

    expect(screen.getByText('Download')).toBeInTheDocument();
    expect(screen.getByText('Copy Path')).toBeInTheDocument();
  });

  it('opens the rename file modal for read/write dataset', async () => {
    renderComponent({ isReadWriteDataset: true });
    await openMenu();
    const renameButton = screen.getByText('Rename');
    await user.click(renameButton);
    // RenameFileModal renders with title "Edit File".
    expect(screen.getByText('Edit File')).toBeInTheDocument();
  });

  it('opens the delete file modal for read/write dataset', async () => {
    renderComponent({ isReadWriteDataset: true });
    await openMenu();
    const deleteButton = screen.getByText('Delete');
    await user.click(deleteButton);
    expect(screen.getByText('Delete File')).toBeInTheDocument();
  });
});
