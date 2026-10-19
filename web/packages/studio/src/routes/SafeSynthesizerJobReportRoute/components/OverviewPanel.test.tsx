// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { triggerDownload } from '@nemo/common/src/utils/file';
import * as safeSynthesizerApi from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import type { GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { OverviewPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/OverviewPanel';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock brand assets icons
vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

// Mock SafeSynthesizerFilesetPreview component
vi.mock('@studio/components/SafeSynthesizerFilesetPreview', () => ({
  SafeSynthesizerFilesetPreview: ({ job }: { job: GenerateJob }) => (
    <div data-testid="fileset-preview">{job.name}</div>
  ),
}));

// Mock triggerDownload utility
vi.mock('@nemo/common/src/utils/file', () => ({
  triggerDownload: vi.fn(),
}));

const mockTriggerDownload = vi.mocked(triggerDownload);

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
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </ThemeProvider>
    );
  };
};

// Helper function to create mock job
const createMockJob = (overrides?: Partial<GenerateJob>): GenerateJob =>
  ({
    id: 'test-job-id',
    name: 'Test Job',
    status: 'COMPLETED',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T01:00:00Z',
    ...overrides,
  }) as GenerateJob;

// Helper function to mock API hooks
const mockApiHooks = (job: GenerateJob, report: string | undefined = undefined) => {
  vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerGetJobSuspense').mockImplementation(
    vi.fn().mockReturnValue({ data: job })
  );
  vi.spyOn(
    safeSynthesizerApi,
    'useSafeSynthesizerDownloadJobResultEvaluationReport'
  ).mockImplementation(vi.fn().mockReturnValue({ data: report }));
};

describe('OverviewPanel', () => {
  beforeEach(() => {
    mockUseParams({ workspace: 'test-workspace' });
  });

  it('renders the panel with correct title and icon', () => {
    const mockJob = createMockJob();
    mockApiHooks(mockJob);

    const testIcon = <svg data-testid="test-icon" />;

    render(<OverviewPanel jobId="test-job-id" title="Test Title" icon={testIcon} />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByTestId('test-icon')).toBeInTheDocument();
  });

  it('renders SafeSynthesizerFilesetPreview with the job', () => {
    const mockJob = createMockJob({ name: 'My Test Job' });
    mockApiHooks(mockJob);

    render(<OverviewPanel jobId="test-job-id" title="Overview" icon={<div />} />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByTestId('fileset-preview')).toBeInTheDocument();
    expect(screen.getByText('My Test Job')).toBeInTheDocument();
  });

  it('shows download button when report is available', () => {
    const mockJob = createMockJob();
    const mockReport = '<html>Report content</html>';
    mockApiHooks(mockJob, mockReport);

    render(<OverviewPanel jobId="test-job-id" title="Overview" icon={<div />} />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByText('Download Report')).toBeInTheDocument();
    expect(screen.getByTestId('download-icon')).toBeInTheDocument();
  });

  it('does not show download button when report is not available', () => {
    const mockJob = createMockJob();
    mockApiHooks(mockJob);

    render(<OverviewPanel jobId="test-job-id" title="Overview" icon={<div />} />, {
      wrapper: createWrapper(),
    });

    expect(screen.queryByText('Download Report')).not.toBeInTheDocument();
  });

  it('triggers download when download button is clicked', async () => {
    const mockJob = createMockJob({ name: 'Test Job' });
    const mockReport = '<html>Report content</html>';
    mockApiHooks(mockJob, mockReport);

    const user = userEvent.setup();

    render(<OverviewPanel jobId="test-job-id" title="Overview" icon={<div />} />, {
      wrapper: createWrapper(),
    });

    const downloadButton = screen.getByText('Download Report');
    await user.click(downloadButton);

    expect(mockTriggerDownload).toHaveBeenCalledWith(mockReport, 'Test Job-report.html');
  });
});
