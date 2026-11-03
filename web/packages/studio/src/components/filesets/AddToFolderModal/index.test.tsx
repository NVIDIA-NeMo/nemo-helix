// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  AddToFolderModal,
  FOLDER_DELETION_WARNING,
} from '@studio/components/filesets/AddToFolderModal';
import type { FileSystemNode } from '@studio/components/FilesTable/utils';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { mockUseParams, mockUseNavigate } from '@studio/tests/util/mockUseParams';
import { render } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the mutation hook
const mockMutateAsync = vi.fn();
const mockIsPending = vi.fn().mockReturnValue(false);
const mockShouldError = vi.fn().mockReturnValue(false);

vi.mock('@studio/api/datasets/useDatasetFilesMove', () => ({
  useDatasetFilesMove: (options?: { onSuccess?: () => void; onError?: (err: Error) => void }) => ({
    mutateAsync: mockMutateAsync.mockImplementation(async () => {
      if (mockShouldError()) {
        options?.onError?.(new Error('Move failed'));
        throw new Error('Move failed');
      }
      options?.onSuccess?.();
    }),
    isPending: mockIsPending(),
  }),
}));

// Mock useDatasetNavigator hook
vi.mock('@studio/hooks/useDatasetNavigator', () => ({
  useDatasetNavigator: () => [],
}));

// Mock data
const MOCK_DATASET_NAME = 'test-dataset';

const mockFiles: FileSystemNode[] = [
  {
    type: 'file',
    path: 'folder1/file1.txt',
    size: 100,
    oid: 'oid1',
  },
  {
    type: 'file',
    path: 'folder1/file2.json',
    size: 200,
    oid: 'oid2',
  },
];

const mockFolderContents: FileSystemNode[] = [
  {
    type: 'directory',
    path: 'folder1/subfolder-a',
    size: 0,
    oid: 'dir1',
    children: {},
  },
  {
    type: 'directory',
    path: 'folder1/subfolder-b',
    size: 0,
    oid: 'dir2',
    children: {},
  },
  {
    type: 'file',
    path: 'folder1/file1.txt',
    size: 100,
    oid: 'oid1',
  },
];

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  selectedItems: mockFiles,
  workspace: DEFAULT_WORKSPACE,
  datasetName: MOCK_DATASET_NAME,
  currentFolder: 'folder1',
  folderContents: mockFolderContents,
  onComplete: vi.fn(),
};

const mockNavigate = vi.fn();

describe('AddToFolderModal', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseParams({
      [ROUTE_PARAMS.workspace]: DEFAULT_WORKSPACE,
    });
    mockUseNavigate(mockNavigate);
    mockMutateAsync.mockResolvedValue({});
    mockIsPending.mockReturnValue(false);
    mockShouldError.mockReturnValue(false);
  });

  describe('Component Rendering', () => {
    it('renders the modal with correct heading', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      expect(await screen.findByRole('heading', { name: 'Move' })).toBeInTheDocument();
    });

    it('renders folder label', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      expect(await screen.findByText('Folder')).toBeInTheDocument();
    });

    it('renders move and cancel buttons', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Move/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
      });
    });

    it('renders select dropdown with placeholder', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = await screen.findByTestId('nv-select-trigger');
      expect(trigger).toBeInTheDocument();
      expect(trigger).toHaveTextContent('Select a folder');
    });
  });

  describe('Form Validation', () => {
    it('disables move button initially when no folder selected', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const moveButton = await screen.findByRole('button', { name: /Move/i });
      await waitFor(() => expect(moveButton).toBeDisabled());
    });
  });

  describe('Folder Options', () => {
    it('shows parent folder option when not at root', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      await waitFor(() => {
        expect(screen.getByRole('option', { name: '.. (parent folder)' })).toBeInTheDocument();
      });
    });

    it('hides parent folder option when at root', async () => {
      const propsAtRoot = {
        ...defaultProps,
        currentFolder: undefined,
      };

      render(<AddToFolderModal {...propsAtRoot} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      // Wait for dropdown to be open, then verify parent folder option is not shown
      await waitFor(() => {
        expect(screen.queryByText('.. (parent folder)')).not.toBeInTheDocument();
      });
    });

    it('shows new folder option', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      await waitFor(() => {
        expect(screen.getByRole('option', { name: 'New Folder' })).toBeInTheDocument();
      });
    });
  });

  describe('New Folder Flow', () => {
    it('shows folder name input when New folder is selected', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const newFolderOption = await screen.findByRole('option', { name: 'New Folder' });
      await user.click(newFolderOption);

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Enter Folder Name')).toBeInTheDocument();
      });
    });

    it('calls moveFiles with new folder path when creating new folder', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const newFolderOption = await screen.findByRole('option', { name: 'New Folder' });
      await user.click(newFolderOption);

      const input = await screen.findByPlaceholderText('Enter Folder Name');
      await user.type(input, 'my-new-folder');

      const moveButton = screen.getByRole('button', { name: /Move/i });
      await user.click(moveButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          workspace: DEFAULT_WORKSPACE,
          name: MOCK_DATASET_NAME,
          filePaths: ['folder1/file1.txt', 'folder1/file2.json'],
          targetFolder: 'folder1/my-new-folder',
        });
      });
    });
  });

  describe('Move Functionality', () => {
    it('calls moveFiles with parent folder when ".." selected', async () => {
      render(<AddToFolderModal {...defaultProps} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const parentOption = await screen.findByRole('option', {
        name: '.. (parent folder)',
      });
      await user.click(parentOption);

      const moveButton = screen.getByRole('button', { name: /Move/i });
      await user.click(moveButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          workspace: DEFAULT_WORKSPACE,
          name: MOCK_DATASET_NAME,
          filePaths: ['folder1/file1.txt', 'folder1/file2.json'],
          targetFolder: '',
        });
      });
    });

    it('calls onComplete callback after successful move', async () => {
      const onCompleteSpy = vi.fn();
      const propsWithCallback = {
        ...defaultProps,
        onComplete: onCompleteSpy,
      };

      render(<AddToFolderModal {...propsWithCallback} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const newFolderOption = await screen.findByRole('option', { name: 'New Folder' });
      await user.click(newFolderOption);

      const input = await screen.findByPlaceholderText('Enter Folder Name');
      await user.type(input, 'test-folder');

      const moveButton = screen.getByRole('button', { name: /Move/i });
      await user.click(moveButton);

      await waitFor(() => {
        expect(onCompleteSpy).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Loading States', () => {
    it('disables cancel button when pending', async () => {
      mockIsPending.mockReturnValue(true);

      render(<AddToFolderModal {...defaultProps} />);

      const cancelButton = await screen.findByRole('button', { name: /cancel/i });
      await waitFor(() => expect(cancelButton).toBeDisabled());
    });

    it('disables move button when pending', async () => {
      mockIsPending.mockReturnValue(true);

      render(<AddToFolderModal {...defaultProps} />);

      const moveButton = await screen.findByRole('button', { name: /Move/i });
      await waitFor(() => expect(moveButton).toBeDisabled());
    });

    it('disables select dropdown when pending', async () => {
      mockIsPending.mockReturnValue(true);

      render(<AddToFolderModal {...defaultProps} />);

      const trigger = await screen.findByTestId('nv-select-trigger');
      await waitFor(() => expect(trigger).toBeDisabled());
    });
  });

  describe('Modal Close Behavior', () => {
    it('calls onClose when cancel button is clicked', async () => {
      const onCloseSpy = vi.fn();
      const propsWithCallback = {
        ...defaultProps,
        onClose: onCloseSpy,
      };

      render(<AddToFolderModal {...propsWithCallback} />);

      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      await user.click(cancelButton);

      expect(onCloseSpy).toHaveBeenCalledTimes(1);
    });
  });

  describe('Folder Deletion Warning', () => {
    const singleFileContents: FileSystemNode[] = [
      {
        type: 'file',
        path: 'folder1/only-file.txt',
        size: 100,
        oid: 'oid1',
      },
    ];

    const propsWithSingleFile = {
      ...defaultProps,
      selectedItems: singleFileContents,
      folderContents: singleFileContents,
    };

    it('shows warning when moving all files to parent folder', async () => {
      render(<AddToFolderModal {...propsWithSingleFile} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const parentOption = await screen.findByRole('option', {
        name: '.. (parent folder)',
      });
      await user.click(parentOption);

      await waitFor(() => {
        expect(screen.getByText(FOLDER_DELETION_WARNING)).toBeInTheDocument();
      });
    });

    it('navigates to parent folder after moving all files out', async () => {
      render(<AddToFolderModal {...propsWithSingleFile} />);

      const trigger = screen.getByTestId('nv-select-trigger');
      await user.click(trigger);

      const parentOption = await screen.findByRole('option', {
        name: '.. (parent folder)',
      });
      await user.click(parentOption);

      const moveButton = screen.getByRole('button', { name: /Move/i });
      await user.click(moveButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith(
          expect.stringContaining(
            `/workspaces/${DEFAULT_WORKSPACE}/filesets/${DEFAULT_WORKSPACE}%2F${MOCK_DATASET_NAME}`
          )
        );
      });
    });
  });
});
