// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatTimeInSeconds, getDifferenceInMilliseconds } from '@nemo/common/src/utils/date';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import * as safeSynthesizerApi from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import type { GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import * as useDatastoreFileContentModule from '@studio/api/datasets/useDatastoreFileContent';
import { JobDetailsPanel } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobDetailsPanel';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the required path params hook
vi.mock('@studio/util/hooks/useRequiredPathParams', () => ({
  useRequiredPathParams: vi.fn(() => ({ safeSynthesizerJobName: 'test-job-id-123' })),
}));

// Mock the date formatting utilities
vi.mock('@studio/util/date', () => ({
  formatDateTime: vi.fn((date: string) => `Formatted: ${date}`),
  formatElapsedTime: vi.fn((start: Date, end: Date) => {
    const diff = end.getTime() - start.getTime();
    return `${Math.floor(diff / 1000)}s`;
  }),
}));

// Mock the util functions
vi.mock('@studio/routes/SafeSynthesizerJobDetailsRoute/util', () => ({
  getFileType: vi.fn(() => 'jsonl'),
  isJobTerminated: vi.fn(
    (status: HelixJobStatus) =>
      status === 'completed' || status === 'error' || status === 'cancelled'
  ),
  getElapsedTime: vi.fn((created_at?: string, resultSummary_created_at?: string) => {
    if (!created_at) return null;
    if (!resultSummary_created_at) return '00:05:30'; // Running job
    return '00:30:00'; // Completed job
  }),
  SAFE_SYNTHESIZER_POLLING_INTERVAL_MS: 5000,
}));

// Mock the toast hook
vi.mock('@nemo/common/src/providers/toast/useToast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  }),
}));

// Note: JobConfigDrawer is no longer mocked - using real component with mocked toast

// Mock parseFilesetUrl utility
vi.mock('@nemo/common/src/components/DatasetFileSelect/utils', () => ({
  parseFilesetUrl: vi.fn((url: string) => {
    if (!url || !url.startsWith('fileset://')) {
      return null;
    }
    const withoutProtocol = url.replace('fileset://', '');
    const parts = withoutProtocol.split('/');
    return {
      workspace: parts[0],
      name: parts[1],
      path: parts.slice(2).join('/'),
    };
  }),
}));

// Mock FilePreview
vi.mock('@studio/components/SafeSynthesizerFilesetPreview/FilePreview', () => ({
  FilePreview: ({
    onClose,
    title,
    isLoading,
    error,
    children,
  }: {
    onClose: () => void;
    title: string;
    isLoading: boolean;
    error?: string;
    children: React.ReactNode;
  }) => (
    <div data-testid="file-preview-panel">
      <div data-testid="preview-heading">{title}</div>
      {isLoading && <div data-testid="preview-loading">Loading...</div>}
      {error && <div data-testid="preview-error">{error}</div>}
      {!isLoading && !error && <div data-testid="preview-content">{children}</div>}
      <button onClick={onClose}>Close Preview</button>
    </div>
  ),
}));

// Mock ScrollTable
vi.mock('@nemo/common/src/components/ScrollTable', () => ({
  ScrollTable: () => <div data-testid="scroll-table">Table Content</div>,
}));

// Mock CodeEditor
vi.mock('@nemo/common/src/components/CodeEditor', () => ({
  CodeEditor: () => <div data-testid="code-editor">Code Content</div>,
  ContentType: {
    JSON: 'json',
    JSONL: 'jsonl',
    YAML: 'yaml',
    JAVASCRIPT: 'javascript',
  },
}));

// Mock StatusBadge
vi.mock('@nemo/common/src/components/StatusBadge', () => ({
  StatusBadge: ({ status }: { status: string }) => <div data-testid="status-badge">{status}</div>,
}));

// Mock toast
vi.mock('@studio/providers/toast/useToast', () => ({
  useToast: vi.fn(() => ({
    error: vi.fn(),
  })),
}));

// Mock brand assets icons
vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

// Test wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider density="standard" theme="light">
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </ThemeProvider>
  );
};

const createdAt = new Date();
createdAt.setFullYear(createdAt.getFullYear() - 1);
const updatedAt = new Date(createdAt);
updatedAt.setHours(updatedAt.getHours() + 1);

// Mock data
const createMockJob = (overrides?: Partial<GenerateJob>): GenerateJob => ({
  id: 'test-job-id-123',
  name: 'Test Safe Synthesizer Job',
  description: 'A test job for synthetic data generation',
  project: 'test-project',
  workspace: 'test-namespace',
  created_at: createdAt.toISOString(),
  updated_at: updatedAt.toISOString(),
  status: 'completed',
  spec: {
    data_source: 'fileset://test-namespace/test-dataset/train.jsonl',
    config: {},
  },
  ownership: {
    created_by: 'test-user@example.com',
  },
  ...overrides,
});

describe('JobDetailsPanel', () => {
  const mockRefetchSourceFile = vi.fn();
  const mockRefetchSyntheticData = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations using vi.spyOn with mockImplementation
    // Note: artifact_url uses fileset:// format as it's parsed by parseFilesetUrl
    vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerListJobResults').mockImplementation(
      vi.fn().mockReturnValue({
        data: {
          data: [
            {},
            { created_at: '2024-01-01T10:30:00.000Z' },
            { artifact_url: 'fileset://test-namespace/results/synthetic_data.csv' },
          ],
        },
      })
    );
    vi.spyOn(useDatastoreFileContentModule, 'useDatastoreFileContent').mockImplementation(
      vi.fn().mockReturnValue({
        data: 'source file content',
        refetch: mockRefetchSourceFile,
      })
    );

    vi.spyOn(
      safeSynthesizerApi,
      'useSafeSynthesizerDownloadJobResultSyntheticData'
    ).mockImplementation(
      vi.fn().mockReturnValue({
        data: undefined,
        refetch: mockRefetchSyntheticData,
      })
    );
  });

  describe('Basic Rendering', () => {
    it('should render job details panel with heading', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Job Details')).toBeInTheDocument();
      expect(screen.getByTestId('play-icon')).toBeInTheDocument();
    });

    it('should display job status with badge', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByTestId('status-badge')).toHaveTextContent('completed');
    });

    it('should display job ID', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Job ID')).toBeInTheDocument();
      expect(screen.getByText('test-job-id-123')).toBeInTheDocument();
    });

    it('should display job name', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Name')).toBeInTheDocument();
      expect(screen.getByText('Test Safe Synthesizer Job')).toBeInTheDocument();
    });

    it('should display job description', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Description')).toBeInTheDocument();
      expect(screen.getByText('A test job for synthetic data generation')).toBeInTheDocument();
    });

    it('should display formatted creation date', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Created')).toBeInTheDocument();
      // The component uses RelativeTime which displays relative text like "last year"
      const timeElement = screen.getByText(/last year/i);
      expect(timeElement).toBeInTheDocument();
      expect(timeElement).toHaveAttribute('datetime', createdAt.toISOString());
    });

    it('should display created by information', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Created by')).toBeInTheDocument();
      expect(screen.getByText('test-user@example.com')).toBeInTheDocument();
    });
  });

  describe('Elapsed Time Display', () => {
    it('should display elapsed time for completed job', () => {
      const job = createMockJob({ status: 'completed' });
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      // Should show elapsed time in HH:MM:SS format
      const diff = getDifferenceInMilliseconds(job.created_at, job.updated_at);
      const elapsedSeconds = diff ? Math.floor(diff / 1000) : undefined;
      const timeString = formatTimeInSeconds(elapsedSeconds);
      expect(screen.getByText(timeString)).toBeInTheDocument();
    });

    it('should calculate elapsed time for running job', async () => {
      const job = createMockJob({ status: 'active' });
      vi.useFakeTimers();
      vi.setSystemTime(new Date(job.created_at!));

      // For running jobs, the results query is disabled, so data should be undefined
      vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerListJobResults').mockImplementation(
        vi.fn().mockReturnValue({
          data: undefined,
        })
      );

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(screen.getByText(formatTimeInSeconds(2))).toBeInTheDocument();
      vi.useRealTimers();
    });

    it('should not display elapsed time when created_at is missing', () => {
      const job = createMockJob({ created_at: undefined });
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByTestId('status-badge')).toBeInTheDocument();
      // No elapsed time displayed
    });
  });

  describe('Data Source Display', () => {
    it('should display data source label and link', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Data Source')).toBeInTheDocument();
      expect(screen.getByText('test-namespace/test-dataset/train.jsonl')).toBeInTheDocument();
    });
  });

  describe('Generation Results Display', () => {
    it('should display generation results when available', () => {
      const job = createMockJob({ status: 'completed' });
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Generation Results')).toBeInTheDocument();
      expect(screen.getByText('synthetic_data.csv')).toBeInTheDocument();
    });

    it('should not display generation results when job is not completed', () => {
      const job = createMockJob({ status: 'active' });
      vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerListJobResults').mockImplementation(
        vi.fn().mockReturnValue({
          data: undefined,
        })
      );
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Generation Results')).toBeInTheDocument();
      expect(screen.queryByText('synthetic_data.csv')).not.toBeInTheDocument();
    });

    it('should make generation results clickable', async () => {
      const job = createMockJob({ status: 'completed' });
      const user = userEvent.setup();

      // Mock the refetch to return a proper response
      mockRefetchSyntheticData.mockResolvedValue({
        data: new Blob(['synthetic data'], { type: 'text/csv' }),
      });

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const resultsLink = screen.getByText('synthetic_data.csv');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(mockRefetchSyntheticData).toHaveBeenCalled();
      });
    });

    it('should open synthetic data preview when clicked', async () => {
      const job = createMockJob({ status: 'completed' });
      const user = userEvent.setup();

      mockRefetchSyntheticData.mockResolvedValue({
        data: new Blob(['synthetic data content'], { type: 'text/csv' }),
      });

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const resultsLink = screen.getByText('synthetic_data.csv');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview-panel')).toBeInTheDocument();
      });
    });
  });

  describe('File Preview Functionality', () => {
    // Note: File preview functionality is only available for Generation Results (synthetic data),
    // not for Data Source. The following tests use Generation Results link to test preview states.

    it('should show loading state in file preview', async () => {
      const job = createMockJob({ status: 'completed' });
      const user = userEvent.setup();

      // Mock refetch to return a promise that never resolves (simulating loading)
      const refetchPromise = new Promise<{ data: Blob | undefined }>(() => {
        // Never resolves - this simulates a loading state
      });
      mockRefetchSyntheticData.mockReturnValue(refetchPromise);

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const resultsLink = screen.getByText('synthetic_data.csv');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(screen.getByTestId('preview-loading')).toBeInTheDocument();
      });
    });

    it('should show error message in file preview', async () => {
      const job = createMockJob({ status: 'completed' });
      const user = userEvent.setup();

      // Mock refetch to return a resolved promise with no data
      mockRefetchSyntheticData.mockResolvedValue({
        data: undefined,
      });

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const resultsLink = screen.getByText('synthetic_data.csv');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(screen.getByTestId('preview-error')).toHaveTextContent(
          'Error fetching synthetic data'
        );
      });
    });

    it('should display correct file content in preview', async () => {
      const job = createMockJob({ status: 'completed' });
      const user = userEvent.setup();

      // Mock refetch to return CSV data as Blob
      mockRefetchSyntheticData.mockResolvedValue({
        data: new Blob(['col1,col2\nval1,val2'], { type: 'text/csv' }),
      });

      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const resultsLink = screen.getByText('synthetic_data.csv');
      await user.click(resultsLink);

      await waitFor(() => {
        expect(screen.getByTestId('preview-content')).toBeInTheDocument();
      });
    });
  });

  describe('Job Config Drawer Interaction', () => {
    it('should display View Job Config button', () => {
      const job = createMockJob();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('View Job Config')).toBeInTheDocument();
      expect(screen.getByTestId('cog-icon')).toBeInTheDocument();
    });

    it('should open job config drawer when button is clicked', async () => {
      const job = createMockJob();
      const user = userEvent.setup();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      const configButton = screen.getByText('View Job Config');
      await user.click(configButton);

      await waitFor(() => {
        expect(screen.getByTestId('job-config-drawer')).toBeInTheDocument();
      });
    });

    it('should close job config drawer when drawer close is triggered', async () => {
      const job = createMockJob();
      const user = userEvent.setup();
      render(<JobDetailsPanel job={job} />, { wrapper: createWrapper() });

      // Open drawer
      const configButton = screen.getByText('View Job Config');
      await user.click(configButton);

      await waitFor(() => {
        expect(screen.getByTestId('job-config-drawer')).toBeInTheDocument();
      });

      // Close drawer using the close button
      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('job-config-drawer')).not.toBeInTheDocument();
      });
    });
  });
});
