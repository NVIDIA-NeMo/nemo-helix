// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvaluateJob, HelixJobStatus } from '@nemo/sdk/generated/evaluator/schema';
import { EvaluationJobBulkDeleteModal } from '@studio/components/evaluation/Jobs/EvaluationJobBulkDeleteModal';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { customRender as render } from '@studio/tests/util/render';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';

const TEST_WORKSPACE = 'test-workspace';

// Mock the delete API
const mockDeleteEvaluateJob = vi.fn();
vi.mock('@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes', async (importOriginal) => {
  const original = await importOriginal();
  return {
    // @ts-expect-error expect issue here with spread
    ...original,
    useEvaluatorDeleteEvaluateJob: vi.fn(() => ({
      mutateAsync: mockDeleteEvaluateJob,
    })),
  };
});

describe('EvaluationJobBulkDeleteModal', () => {
  const mockJobs: EvaluateJob[] = [
    {
      id: 'job-1',
      name: 'job-1',
      status: HelixJobStatus.completed,
      created_at: '2024-01-01T00:00:00Z',
      spec: {
        metrics: [],
        dataset: [],
      },
      custom_fields: {},
    },
    {
      id: 'job-2',
      name: 'job-2',
      status: HelixJobStatus.active,
      created_at: '2024-01-02T00:00:00Z',
      spec: {
        metrics: [],
        dataset: [],
      },
      custom_fields: {},
    },
  ];

  const mockOnConfirmSuccess = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockDeleteEvaluateJob.mockResolvedValue(undefined);
    mockUseParams({
      [ROUTE_PARAMS.workspace]: TEST_WORKSPACE,
    });
  });

  describe('Rendering', () => {
    it('should render trigger button with correct text', () => {
      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={mockJobs}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      expect(screen.getByTestId('bulk-delete-modal-trigger-button')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });

    it('should show modal when trigger is clicked', async () => {
      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={mockJobs}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      expect(await screen.findByText('Delete 2 Jobs')).toBeInTheDocument();
      expect(screen.getByText('Are you sure you want to delete this?')).toBeInTheDocument();
    });

    it('should show singular form for single job', async () => {
      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={[mockJobs[0]]}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      expect(await screen.findByText('Delete 1 Job')).toBeInTheDocument();
    });
  });

  describe('Modal Actions', () => {
    it('should close modal when cancel is clicked', async () => {
      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={mockJobs}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      // Open modal
      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      await waitFor(() => {
        expect(screen.getByText('Delete 2 Jobs')).toBeInTheDocument();
      });

      // Click cancel
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      fireEvent.click(cancelButton);

      await waitFor(() => {
        expect(screen.queryByText('Delete 2 Jobs')).not.toBeInTheDocument();
      });
    });

    it('should call onConfirmSuccess and close modal when delete is confirmed', async () => {
      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={mockJobs}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      // Open modal
      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      expect(await screen.findByText('Delete 2 Jobs')).toBeInTheDocument();

      // Click delete
      const deleteButton = within(screen.getByRole('dialog')).getByRole('button', {
        name: 'Delete',
      });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(mockDeleteEvaluateJob).toHaveBeenCalledTimes(2);
      });
      await waitFor(() => {
        expect(mockDeleteEvaluateJob).toHaveBeenCalledWith({
          workspace: TEST_WORKSPACE,
          name: 'job-1',
        });
      });
      await waitFor(() => {
        expect(mockDeleteEvaluateJob).toHaveBeenCalledWith({
          workspace: TEST_WORKSPACE,
          name: 'job-2',
        });
      });
      await waitFor(() => {
        expect(mockOnConfirmSuccess).toHaveBeenCalled();
      });

      // Modal should be closed
      await waitFor(() => {
        expect(screen.queryByText('Delete 2 Jobs')).not.toBeInTheDocument();
      });
    });

    it('should handle deletion errors gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockDeleteEvaluateJob.mockRejectedValue(new Error('Delete failed'));

      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={mockJobs}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      // Open modal
      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      expect(await screen.findByText('Delete 2 Jobs')).toBeInTheDocument();

      // Click delete
      const deleteButton = within(screen.getByRole('dialog')).getByRole('button', {
        name: 'Delete',
      });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          'Failed to delete evaluation jobs',
          expect.any(Error)
        );
      });
      await waitFor(() => {
        expect(mockOnConfirmSuccess).not.toHaveBeenCalled();
      });

      consoleSpy.mockRestore();
    });

    it('should filter out jobs without valid names', async () => {
      const jobsWithInvalidNames = [
        { ...mockJobs[0], name: undefined as unknown as string },
        mockJobs[1],
        { ...mockJobs[0], name: 'job-3' },
      ];

      render(
        <EvaluationJobBulkDeleteModal
          selectedJobs={jobsWithInvalidNames}
          onConfirmSuccess={mockOnConfirmSuccess}
        />
      );

      // Open modal
      const triggerButton = screen.getByTestId('bulk-delete-modal-trigger-button');
      fireEvent.click(triggerButton);

      expect(await screen.findByText('Delete 3 Jobs')).toBeInTheDocument();

      // Click delete
      const deleteButton = within(screen.getByRole('dialog')).getByRole('button', {
        name: 'Delete',
      });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        // Should only call delete for jobs with valid names
        expect(mockDeleteEvaluateJob).toHaveBeenCalledTimes(2);
      });
      await waitFor(() => {
        expect(mockDeleteEvaluateJob).toHaveBeenCalledWith({
          workspace: TEST_WORKSPACE,
          name: 'job-2',
        });
      });
      await waitFor(() => {
        expect(mockDeleteEvaluateJob).toHaveBeenCalledWith({
          workspace: TEST_WORKSPACE,
          name: 'job-3',
        });
      });
      await waitFor(() => {
        expect(mockOnConfirmSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Modal State Management', () => {
    it('should handle empty selected jobs array', () => {
      render(
        <EvaluationJobBulkDeleteModal selectedJobs={[]} onConfirmSuccess={mockOnConfirmSuccess} />
      );

      expect(screen.getByTestId('bulk-delete-modal-trigger-button')).toBeInTheDocument();
      expect(screen.getByText('Delete')).toBeInTheDocument();
    });
  });
});
