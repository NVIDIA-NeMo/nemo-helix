// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@nvidia/foundations-react-core';
import { ReportSummaryPanel } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/ReportSummaryPanel';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

// Mock React Router hooks
const mockNavigate = vi.fn();
vi.mock('react-router', async () => {
  const actual = await vi.importActual<typeof import('react-router')>('react-router');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock useWorkspaceFromPath hook
vi.mock('@studio/hooks/useWorkspaceFromPath', () => ({
  useWorkspaceFromPath: vi.fn(() => 'test-project'),
}));

// Mock the ScoreGauge component
vi.mock('@nemo/common/src/components/ScoreGauge', () => ({
  ScoreGauge: ({ score, size }: { score?: number; size: string }) => (
    <div data-testid="score-gauge">
      <div data-testid="gauge-score">{score}</div>
      <div data-testid="gauge-size">{size}</div>
    </div>
  ),
}));

// Mock brand assets icons
// Mock the util constants
vi.mock('@studio/routes/GenerateJobRoute/util', () => ({
  SAFE_SYNTHESIZER_POLLING_INTERVAL_MS: 5000,
}));

// Test wrapper
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter>
      <ThemeProvider density="standard" theme="light">
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </ThemeProvider>
    </MemoryRouter>
  );
};

describe('ReportSummaryPanel', () => {
  const mockJobResultSummary = {
    synthetic_data_quality_score: 8.5,
    data_privacy_score: 7.3,
    timing: {
      total_time_sec: 100,
      pii_replacer_time_sec: 20,
      training_time_sec: 40,
      generation_time_sec: 30,
      evaluation_time_sec: 10,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('should render report summary panel with heading', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={mockJobResultSummary} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('Report Summary')).toBeInTheDocument();
    });

    it('should render Quality (SQS) and Privacy (DPS) labels', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={mockJobResultSummary} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('Quality (SQS)')).toBeInTheDocument();
      expect(screen.getByText('Privacy (DPS)')).toBeInTheDocument();
    });

    it('should render two score gauges', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={mockJobResultSummary} />, {
        wrapper: createWrapper(),
      });

      const gauges = screen.getAllByTestId('score-gauge');
      expect(gauges).toHaveLength(2);
    });
  });

  describe('View Report Button', () => {
    it('should render view report button when jobResultSummary is present', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={mockJobResultSummary} />, {
        wrapper: createWrapper(),
      });

      expect(screen.getByText('View Report')).toBeInTheDocument();
    });

    it('should not render view report button when jobResultSummary is not present', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={undefined} />, {
        wrapper: createWrapper(),
      });

      expect(screen.queryByText('View Report')).not.toBeInTheDocument();
    });

    it('should navigate to report page when view report button is clicked', () => {
      render(<ReportSummaryPanel jobId="test-job-id" jobResultSummary={mockJobResultSummary} />, {
        wrapper: createWrapper(),
      });

      const viewReportButton = screen.getByText('View Report');
      fireEvent.click(viewReportButton);
      expect(mockNavigate).toHaveBeenCalledWith(
        '/workspaces/test-project/safe-synthesizer/job/test-job-id/report'
      );
    });
  });
});
