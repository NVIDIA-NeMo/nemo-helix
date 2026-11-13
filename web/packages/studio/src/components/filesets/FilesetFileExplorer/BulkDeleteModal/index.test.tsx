// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BulkDeleteModal } from '@studio/components/filesets/FilesetFileExplorer/BulkDeleteModal';
import { extractFilePathsFromDirectory } from '@studio/components/filesets/FilesetFileExplorer/extractFilePathsFromDirectory';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { render } from '@studio/tests/util/render';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the mutation hook
const mockMutateAsync = vi.fn();

vi.mock('@studio/api/datasets/useDatasetFilesDelete', () => ({
  useDatasetFilesDelete: () => ({
    mutateAsync: mockMutateAsync,
  }),
}));

// Mock data
const mockFiles: FileSystemNode[] = [
  {
    type: 'file',
    path: 'file1.txt',
    size: 100,
    oid: 'oid1',
  },
  {
    type: 'file',
    path: 'file2.json',
    size: 200,
    oid: 'oid2',
  },
];

const mockDirectoryWithFiles: FileSystemNode = {
  type: 'directory',
  path: 'folder1',
  size: 0,
  oid: 'dir1',
  children: {
    subfolder: {
      type: 'directory',
      path: 'folder1/subfolder',
      size: 0,
      oid: 'dir2',
      children: {
        'nested-file.txt': {
          type: 'file',
          path: 'folder1/subfolder/nested-file.txt',
          size: 150,
          oid: 'oid3',
        },
      },
    },
    'direct-file.csv': {
      type: 'file',
      path: 'folder1/direct-file.csv',
      size: 300,
      oid: 'oid4',
    },
  },
};

const mockDirectories: FileSystemNode[] = [mockDirectoryWithFiles];

const defaultProps = {
  selectedItems: [...mockFiles, ...mockDirectories],
  workspace: 'test-workspace',
  datasetName: 'test-dataset',
  onConfirmDelete: vi.fn(),
};

const getTriggerButton = () => screen.getByTestId('bulk-delete-modal-trigger-button');
const getDialog = () => screen.getByRole('dialog');

describe('BulkDeleteModal', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    mockMutateAsync.mockResolvedValue({});
  });

  describe('Component Rendering', () => {
    it('renders the modal trigger button', () => {
      render(<BulkDeleteModal {...defaultProps} />);

      const triggerButton = getTriggerButton();
      expect(triggerButton).toBeInTheDocument();
      expect(triggerButton).toHaveTextContent('Delete');
    });

    it('opens modal when trigger button is clicked', async () => {
      render(<BulkDeleteModal {...defaultProps} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      expect(screen.getByText('Delete 3 Items')).toBeInTheDocument();
      expect(screen.getByText('Are you sure you want to delete this?')).toBeInTheDocument();
    });

    it('shows correct item count in modal heading', async () => {
      const propsWithSingleItem = {
        ...defaultProps,
        selectedItems: [mockFiles[0]],
      };

      render(<BulkDeleteModal {...propsWithSingleItem} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      expect(screen.getByText('Delete 1 Item')).toBeInTheDocument();
    });

    it('renders cancel and delete buttons in modal', async () => {
      render(<BulkDeleteModal {...defaultProps} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const dialog = getDialog();
      expect(within(dialog).getByRole('button', { name: /cancel/i })).toBeInTheDocument();
      expect(within(dialog).getByRole('button', { name: /^delete$/i })).toBeInTheDocument();
    });
  });

  describe('File Path Extraction', () => {
    it('extracts file path from a file node', () => {
      const fileNode: FileSystemNode = {
        type: 'file',
        path: 'test-file.txt',
        size: 100,
        oid: 'oid1',
      };

      const result = extractFilePathsFromDirectory(fileNode);
      expect(result).toEqual(['test-file.txt']);
    });

    it('extracts all file paths from a directory recursively', () => {
      const result = extractFilePathsFromDirectory(mockDirectoryWithFiles);

      expect(result).toContain('folder1/subfolder/nested-file.txt');
      expect(result).toContain('folder1/direct-file.csv');
      expect(result).toHaveLength(2);
    });

    it('handles empty directory', () => {
      const emptyDirectory: FileSystemNode = {
        type: 'directory',
        path: 'empty-folder',
        size: 0,
        oid: 'empty',
        children: {},
      };

      const result = extractFilePathsFromDirectory(emptyDirectory);
      expect(result).toEqual([]);
    });

    it('handles directory without children property', () => {
      const directoryWithoutChildren: FileSystemNode = {
        type: 'directory',
        path: 'folder-no-children',
        size: 0,
        oid: 'no-children',
      } as FileSystemNode;

      const result = extractFilePathsFromDirectory(directoryWithoutChildren);
      expect(result).toEqual([]);
    });
  });

  describe('Delete Functionality', () => {
    it('calls mutateAsync with correct parameters when deleting files only', async () => {
      const propsWithFilesOnly = {
        ...defaultProps,
        selectedItems: mockFiles,
      };

      render(<BulkDeleteModal {...propsWithFilesOnly} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          workspace: 'test-workspace',
          datasetName: 'test-dataset',
          paths: ['file1.txt', 'file2.json'],
        });
      });
    });

    it('calls mutateAsync with correct parameters when deleting directories only', async () => {
      const propsWithDirectoriesOnly = {
        ...defaultProps,
        selectedItems: mockDirectories,
      };

      render(<BulkDeleteModal {...propsWithDirectoriesOnly} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          workspace: 'test-workspace',
          datasetName: 'test-dataset',
          paths: ['folder1/subfolder/nested-file.txt', 'folder1/direct-file.csv'],
        });
      });
    });

    it('calls mutateAsync with combined file paths when deleting mixed items', async () => {
      render(<BulkDeleteModal {...defaultProps} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          workspace: 'test-workspace',
          datasetName: 'test-dataset',
          paths: [
            'file1.txt',
            'file2.json',
            'folder1/subfolder/nested-file.txt',
            'folder1/direct-file.csv',
          ],
        });
      });
    });

    it('does not call mutateAsync when no items are selected', async () => {
      const propsWithNoItems = {
        ...defaultProps,
        selectedItems: [],
      };

      render(<BulkDeleteModal {...propsWithNoItems} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(mockMutateAsync).not.toHaveBeenCalled();
      });
    });

    it('calls onConfirmDelete callback after successful deletion', async () => {
      const onConfirmDeleteSpy = vi.fn();
      const propsWithCallback = {
        ...defaultProps,
        onConfirmDelete: onConfirmDeleteSpy,
      };

      render(<BulkDeleteModal {...propsWithCallback} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(onConfirmDeleteSpy).toHaveBeenCalledTimes(1);
      });
    });

    it('disables Delete but keeps Cancel enabled while deletion is in progress', async () => {
      let resolveDelete!: () => void;
      mockMutateAsync.mockReturnValue(
        new Promise<void>((resolve) => {
          resolveDelete = resolve;
        })
      );

      render(<BulkDeleteModal {...defaultProps} />);
      await user.click(getTriggerButton());

      const dialog = getDialog();

      // Don't await — the delete is intentionally left pending
      const clickPromise = user.click(within(dialog).getByRole('button', { name: /^delete$/i }));

      await waitFor(() => {
        expect(within(dialog).getByRole('button', { name: /delete/i })).toBeDisabled();
        expect(within(dialog).getByRole('button', { name: /cancel/i })).toBeEnabled();
      });

      // Resolve to allow the modal to close and avoid act() warnings
      resolveDelete();
      await clickPromise;
    });

    it('closes modal after successful deletion', async () => {
      render(<BulkDeleteModal {...defaultProps} />);

      const triggerButton = getTriggerButton();
      await user.click(triggerButton);

      // Verify modal is open
      expect(screen.getByText('Delete 3 Items')).toBeInTheDocument();

      const deleteButton = within(getDialog()).getByRole('button', { name: /^delete$/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(screen.queryByText('Delete 3 Items')).not.toBeInTheDocument();
      });
    });
  });
});
