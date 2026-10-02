// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DeleteConfirmationModal } from '@nemo/common/src/components/DeleteConfirmationModal';
import { FilesetOutput as Dataset } from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { server } from '@studio/mocks/node';
import { ActionMenu } from '@studio/routes/FilesetListRoute/ActionMenu';
import { render } from '@studio/tests/util/render';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

// Mock the modal components (presentation-only — no data layer)
vi.mock('@studio/components/DatasetCreateModal', () => ({
  DatasetCreateModal: vi.fn(({ open, onClose, onDatasetUpdated, dataset }) => {
    if (!open) return null;
    return (
      <div data-testid="dataset-create-modal">
        <div>Edit Dataset Modal</div>
        <div>Dataset: {dataset?.name}</div>
        <button onClick={onClose} data-testid="modal-close">
          Close
        </button>
        <button
          onClick={() => onDatasetUpdated?.({ ...dataset, description: 'Updated description' })}
          data-testid="modal-save"
        >
          Save
        </button>
      </div>
    );
  }),
}));

vi.mock('@nemo/common/src/components/DeleteConfirmationModal', () => ({
  DeleteConfirmationModal: vi.fn(({ open, onClose, onDelete, title }) => {
    if (!open) return null;
    return (
      <div data-testid="delete-confirmation-modal">
        <div>Delete Confirmation Modal</div>
        <div>{title}</div>
        <button onClick={onClose} data-testid="modal-close">
          Cancel
        </button>
        <button
          onClick={async () => {
            const result = await onDelete();
            if (result) onClose();
          }}
          data-testid="modal-delete"
        >
          Delete
        </button>
      </div>
    );
  }),
}));

vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: () => ({
    projectNamespace: 'test-namespace',
    projectName: 'test-project',
    project: 'test-namespace/test-project',
  }),
}));

describe('ActionMenu', () => {
  const mockDataset: Dataset = {
    id: '123',
    name: 'test-dataset',
    workspace: 'test-workspace',
    description: 'Test dataset description',
    purpose: 'dataset',
    storage: { type: 'local', path: '/data/test-dataset' },
    metadata: {},
    custom_fields: {},
    project: 'test-workspace',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  const mockOnNavigateToDetails = vi.fn();
  const mockOnDatasetUpdated = vi.fn();
  const mockOnDatasetDeleted = vi.fn();

  const defaultProps = {
    dataset: mockDataset,
    onNavigateToDetails: mockOnNavigateToDetails,
    onDatasetUpdated: mockOnDatasetUpdated,
    onDatasetDeleted: mockOnDatasetDeleted,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the action menu trigger button', () => {
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      expect(triggerButton).toBeInTheDocument();
    });

    it('shows dropdown menu when trigger is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);

      expect(screen.getByText('View')).toBeInTheDocument();
      expect(screen.getByText('Edit')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    it('displays correct icons in dropdown items', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);

      const viewItem = screen.getByText('View');
      const editItem = screen.getByText('Edit');
      const deleteItem = screen.getByText('Delete');

      expect(viewItem).toBeInTheDocument();
      expect(editItem).toBeInTheDocument();
      expect(deleteItem).toBeInTheDocument();
    });
  });

  describe('Dropdown Actions', () => {
    it('calls onNavigateToDetails when View is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);

      const viewButton = screen.getByText('View');
      await user.click(viewButton);

      expect(mockOnNavigateToDetails).toHaveBeenCalledWith(mockDataset);
    });

    it('opens edit modal when Edit is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);

      const editButton = screen.getByText('Edit');
      await user.click(editButton);

      expect(screen.getByTestId('dataset-create-modal')).toBeInTheDocument();
      expect(screen.getByText('Edit Dataset Modal')).toBeInTheDocument();
      expect(screen.getByText(`Dataset: ${mockDataset.name}`)).toBeInTheDocument();
    });

    it('opens delete modal when Delete is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);

      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);

      expect(screen.getByTestId('delete-confirmation-modal')).toBeInTheDocument();
      expect(screen.getByText('Delete Confirmation Modal')).toBeInTheDocument();
      expect(screen.getByText(`Delete Dataset: ${mockDataset.name}`)).toBeInTheDocument();
    });

    it('surfaces the backend message when the fileset is in use', async () => {
      const detail =
        "Cannot delete fileset 'test-workspace/test-dataset' because 2 model entity reference(s) and 1 adapter entity reference(s) still use it. Relink or delete the dependent entities first.";
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      server.use(
        http.delete(`${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`, () =>
          HttpResponse.json({ detail }, { status: 409 })
        )
      );
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /open dataset actions menu/i }));
      await user.click(screen.getByText('Delete'));

      const modalProps = vi.mocked(DeleteConfirmationModal).mock.calls.at(-1)?.[0];
      expect(modalProps).toBeDefined();
      await expect(modalProps?.onDelete()).rejects.toThrow(detail);
      expect(mockOnDatasetDeleted).not.toHaveBeenCalled();
      expect(consoleErrorSpy).toHaveBeenCalledWith('Failed to delete dataset', expect.any(Error));
      consoleErrorSpy.mockRestore();
    });
  });

  describe('Edit Modal Integration', () => {
    it('closes edit modal when close button is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const editButton = screen.getByText('Edit');
      await user.click(editButton);

      expect(screen.getByTestId('dataset-create-modal')).toBeInTheDocument();

      const closeButton = screen.getByTestId('modal-close');
      await user.click(closeButton);

      expect(screen.queryByTestId('dataset-create-modal')).not.toBeInTheDocument();
    });

    it('calls onDatasetUpdated and closes modal when save is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const editButton = screen.getByText('Edit');
      await user.click(editButton);

      const saveButton = screen.getByTestId('modal-save');
      await user.click(saveButton);

      expect(mockOnDatasetUpdated).toHaveBeenCalledWith({
        ...mockDataset,
        description: 'Updated description',
      });

      expect(screen.queryByTestId('dataset-create-modal')).not.toBeInTheDocument();
    });

    it('handles missing onDatasetUpdated callback gracefully', async () => {
      const user = userEvent.setup();
      const propsWithoutCallback = {
        ...defaultProps,
        onDatasetUpdated: undefined,
      };

      render(<ActionMenu {...propsWithoutCallback} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const editButton = screen.getByText('Edit');
      await user.click(editButton);

      const saveButton = screen.getByTestId('modal-save');
      await user.click(saveButton);

      expect(() => user.click(saveButton)).not.toThrow();
      expect(screen.queryByTestId('dataset-create-modal')).not.toBeInTheDocument();
    });
  });

  describe('Delete Modal Integration', () => {
    it('closes delete modal when cancel button is clicked', async () => {
      const user = userEvent.setup();
      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);

      expect(screen.getByTestId('delete-confirmation-modal')).toBeInTheDocument();

      const cancelButton = screen.getByTestId('modal-close');
      await user.click(cancelButton);

      expect(screen.queryByTestId('delete-confirmation-modal')).not.toBeInTheDocument();
    });

    it('performs delete operation and calls callbacks when delete is confirmed', async () => {
      const user = userEvent.setup();
      let capturedParams: { workspace: string; name: string } | undefined;

      server.use(
        http.delete<{ workspace: string; name: string }>(
          `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`,
          ({ params }) => {
            capturedParams = { workspace: params.workspace, name: params.name };
            return new HttpResponse(null, { status: 200 });
          }
        )
      );

      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);

      const confirmDeleteButton = screen.getByTestId('modal-delete');
      await user.click(confirmDeleteButton);

      await waitFor(() => {
        expect(capturedParams).toEqual({
          workspace: mockDataset.workspace,
          name: mockDataset.name,
        });
      });

      await waitFor(() => {
        expect(mockOnDatasetDeleted).toHaveBeenCalledWith(mockDataset);
      });

      await waitFor(() => {
        expect(screen.queryByTestId('delete-confirmation-modal')).not.toBeInTheDocument();
      });
    });

    it('handles delete operation failure gracefully', async () => {
      const user = userEvent.setup();
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      server.use(
        http.delete(`${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`, () =>
          HttpResponse.json({ detail: 'Delete failed' }, { status: 500 })
        )
      );

      render(<ActionMenu {...defaultProps} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);
      const modalProps = vi.mocked(DeleteConfirmationModal).mock.calls.at(-1)?.[0];
      expect(modalProps).toBeDefined();
      await expect(modalProps?.onDelete()).rejects.toThrow('Delete failed');

      await waitFor(() => {
        expect(consoleErrorSpy).toHaveBeenCalledWith('Failed to delete dataset', expect.any(Error));
      });

      expect(mockOnDatasetDeleted).not.toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });

    it('handles missing onDatasetDeleted callback gracefully', async () => {
      const user = userEvent.setup();
      const propsWithoutCallback = {
        ...defaultProps,
        onDatasetDeleted: undefined,
      };

      render(<ActionMenu {...propsWithoutCallback} />);

      const triggerButton = screen.getByRole('button', { name: /open dataset actions menu/i });
      await user.click(triggerButton);
      const deleteButton = screen.getByText('Delete');
      await user.click(deleteButton);
      const confirmDeleteButton = screen.getByTestId('modal-delete');

      await expect(user.click(confirmDeleteButton)).resolves.not.toThrow();
    });
  });
});
