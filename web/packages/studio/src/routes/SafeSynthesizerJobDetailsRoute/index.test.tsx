// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobLogs } from '@nemo/common/src/hooks/useJobLogs';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import * as safeSynthesizerApi from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import {
  type GenerateJob,
  type SafeSynthesizerSummary,
} from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { GenerateJobDetailsRoute } from '@studio/routes/SafeSynthesizerJobDetailsRoute';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';

// Mock dependencies
vi.mock('@studio/hooks/useWorkspaceFromPath');
vi.mock('@nemo/common/src/hooks/useJobLogs');
vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs');
vi.mock('@studio/routes/utils');

// Mock SafeSynthesizerNavigation component
vi.mock('@studio/components/SafeSynthesizerNavigation', () => ({
  SafeSynthesizerNavigation: ({ selected, jobName }: { selected: string; jobName: string }) => (
    <div data-testid="safe-synthesizer-navigation">
      <span data-testid="navigation-selected">{selected}</span>
      <span data-testid="navigation-job-id">{jobName}</span>
    </div>
  ),
}));

// Mock JobDetailsPanel component
vi.mock('@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobDetailsPanel', () => ({
  JobDetailsPanel: ({ job, errorMessage }: { job: GenerateJob; errorMessage?: string }) => (
    <div data-testid="job-details-panel">
      <span data-testid="panel-job-id">{job.id}</span>
      <span data-testid="panel-job-name">{job.name}</span>
      <span data-testid="panel-job-status">{job.status}</span>
      {errorMessage && <span data-testid="job-details-error-message">{errorMessage}</span>}
    </div>
  ),
}));

// Mock ReportSummaryPanel component
vi.mock('@studio/routes/SafeSynthesizerJobDetailsRoute/components/ReportSummaryPanel', () => ({
  ReportSummaryPanel: ({
    jobId,
    jobResultSummary,
  }: {
    jobId: string;
    jobResultSummary?: SafeSynthesizerSummary;
  }) => (
    <div data-testid="report-summary-panel">
      <span data-testid="panel-job-id">{jobId}</span>
      <span data-testid="panel-has-summary">{jobResultSummary ? 'has-summary' : 'no-summary'}</span>
    </div>
  ),
}));

// Mock ProgressSection component
vi.mock('@studio/routes/SafeSynthesizerJobDetailsRoute/components/ProgressSection', () => ({
  ProgressSection: ({ jobId }: { jobId: string }) => (
    <div data-testid="progress-section">
      <span data-testid="section-job-id">{jobId}</span>
    </div>
  ),
}));

const mockuseWorkspaceFromPath = vi.mocked(useWorkspaceFromPath);
const mockUseJobLogs = vi.mocked(useJobLogs);
const mockUseBreadcrumbs = vi.mocked(useBreadcrumbs);

// Helper function to create test wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <MemoryRouter
            initialEntries={['/projects/test-project/safe-synthesizer/jobs/test-job-123/summary']}
          >
            <Routes>
              <Route
                path="/projects/:project/safe-synthesizer/jobs/:safeSynthesizerJobName/summary"
                element={children}
              />
            </Routes>
          </MemoryRouter>
        </QueryClientProvider>
      </ThemeProvider>
    );
  };
};

// Helper function to create mock job
const createMockJob = (overrides?: Partial<GenerateJob>): GenerateJob =>
  ({
    id: 'test-job-123',
    name: 'Test Job',
    status: HelixJobStatus.completed,
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T01:00:00Z',
    spec: {
      config: {
        data_source: {
          type: 'file',
          path: '/path/to/data.csv',
        },
      },
    },
    ...overrides,
  }) as GenerateJob;

// Helper function to mock API hooks
const mockApiHooks = (
  job: GenerateJob,
  summary: SafeSynthesizerSummary | undefined = undefined
) => {
  vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerGetJobSuspense').mockImplementation(
    vi.fn().mockReturnValue({ data: job })
  );
  vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerDownloadJobResultSummary').mockImplementation(
    vi.fn().mockReturnValue({ data: summary })
  );
};

describe('GenerateJobDetailsRoute - Feature Flag', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('should be defined when feature flag is enabled', async () => {
    // Mock the environment constant to enable the component
    vi.doMock('@studio/constants/environment', () => ({
      SAFE_SYNTHESIZER_ENABLED: true,
    }));

    const module = await import('./index');
    expect(module.GenerateJobDetailsRoute).toBeDefined();
    expect(module.GenerateJobDetailsRoute).not.toBeNull();
  });

  it('should be null when feature flag is disabled', async () => {
    // Mock the environment constant to disable the component
    vi.doMock('@studio/constants/environment', () => ({
      SAFE_SYNTHESIZER_ENABLED: false,
    }));

    const module = await import('./index');
    expect(module.GenerateJobDetailsRoute).toBeNull();
  });
});

describe('GenerateJobDetailsRoute - Rendering', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockuseWorkspaceFromPath.mockReturnValue('test-workspace');
    mockUseJobLogs.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      total: 0,
      loadProgress: null,
      refetch: vi.fn(),
    });
    mockUseBreadcrumbs.mockReturnValue({
      breadcrumbs: [],
      setBreadcrumbs: vi.fn(),
    });
  });

  it('should render all main components', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    mockApiHooks(mockJob);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    // Verify all main components are rendered
    expect(screen.getByTestId('safe-synthesizer-navigation')).toBeInTheDocument();
    expect(screen.getByTestId('job-details-panel')).toBeInTheDocument();
    expect(screen.getByTestId('report-summary-panel')).toBeInTheDocument();
    expect(screen.getByTestId('progress-section')).toBeInTheDocument();
  });

  it('should render navigation with correct selection', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    mockApiHooks(mockJob);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    expect(screen.getByTestId('navigation-selected')).toHaveTextContent('summary');
    expect(screen.getByTestId('navigation-job-id')).toHaveTextContent('test-job-123');
  });

  it('should not fetch summary for running jobs', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.active });
    mockApiHooks(mockJob);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    const reportPanel = screen.getByTestId('report-summary-panel');
    expect(within(reportPanel).getByTestId('panel-has-summary')).toHaveTextContent('no-summary');
  });

  it('should fetch and display summary for completed jobs', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.completed });
    const mockSummary = {
      synthetic_data_quality_score: 8.5,
      data_privacy_score: 7.3,
      timing: {
        total_time_sec: 100,
      },
    } as SafeSynthesizerSummary;
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    const reportPanel = screen.getByTestId('report-summary-panel');
    expect(within(reportPanel).getByTestId('panel-has-summary')).toHaveTextContent('has-summary');
  });

  it('should not fetch summary for error jobs', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.error });
    mockApiHooks(mockJob);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    const reportPanel = screen.getByTestId('report-summary-panel');
    expect(within(reportPanel).getByTestId('panel-has-summary')).toHaveTextContent('no-summary');
  });

  it('should set up breadcrumbs correctly', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    mockApiHooks(mockJob);

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    // Verify breadcrumbs were called with correct items
    expect(mockUseBreadcrumbs).toHaveBeenCalledWith(
      expect.objectContaining({
        items: expect.arrayContaining([
          expect.objectContaining({
            slotLabel: 'Safe Synthesizer',
          }),
          expect.objectContaining({
            slotLabel: 'Job Details',
          }),
        ]),
      })
    );
  });

  it('should extract and display error message from logs when job status is error', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.error });
    mockApiHooks(mockJob);

    // Mock logs with JSON error message
    mockUseJobLogs.mockReturnValue({
      data: [
        {
          timestamp: '2025-01-01T00:00:00Z',
          job: 'test-job-123',
          job_step: 'step1',
          job_task: 'task1',
          message: 'Regular log message',
        },
        {
          timestamp: '2025-01-01T00:01:00Z',
          job: 'test-job-123',
          job_step: 'step2',
          job_task: 'task2',
          message: JSON.stringify({
            level: 'ERROR',
            message: 'An error occurred during processing',
          }),
        },
        {
          timestamp: '2025-01-01T00:02:00Z',
          job: 'test-job-123',
          job_step: 'step3',
          job_task: 'task3',
          message: 'Another log message',
        },
      ],
      isLoading: false,
      error: null,
      total: 3,
      loadProgress: null,
      refetch: vi.fn(),
    });

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    // Verify error message is displayed
    const errorMessage = screen.getByTestId('job-details-error-message');
    expect(errorMessage).toHaveTextContent('An error occurred during processing');
  });

  it('should not display error message when job status is not error', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.completed });
    mockApiHooks(mockJob);

    // Mock logs with JSON error message (should be ignored)
    mockUseJobLogs.mockReturnValue({
      data: [
        {
          timestamp: '2025-01-01T00:01:00Z',
          job: 'test-job-123',
          job_step: 'step2',
          job_task: 'task2',
          message: JSON.stringify({
            level: 'ERROR',
            message: 'An error occurred during processing',
          }),
        },
      ],
      isLoading: false,
      error: null,
      total: 1,
      loadProgress: null,
      refetch: vi.fn(),
    });

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    // Verify error message is not displayed
    expect(screen.queryByTestId('job-details-error-message')).not.toBeInTheDocument();
  });

  it('should not display error message when no ERROR level logs exist', () => {
    if (!GenerateJobDetailsRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({ status: HelixJobStatus.error });
    mockApiHooks(mockJob);

    // Mock logs without ERROR level
    mockUseJobLogs.mockReturnValue({
      data: [
        {
          timestamp: '2025-01-01T00:00:00Z',
          job: 'test-job-123',
          job_step: 'step1',
          job_task: 'task1',
          message: 'Regular log message',
        },
        {
          timestamp: '2025-01-01T00:01:00Z',
          job: 'test-job-123',
          job_step: 'step2',
          job_task: 'task2',
          message: JSON.stringify({ level: 'INFO', message: 'Just an info message' }),
        },
      ],
      isLoading: false,
      error: null,
      total: 2,
      loadProgress: null,
      refetch: vi.fn(),
    });

    render(<GenerateJobDetailsRoute />, { wrapper: createWrapper() });

    // Verify error message is not displayed
    expect(screen.queryByTestId('job-details-error-message')).not.toBeInTheDocument();
  });
});
