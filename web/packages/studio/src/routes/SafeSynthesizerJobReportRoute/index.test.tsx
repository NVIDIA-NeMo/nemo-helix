// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as safeSynthesizerApi from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import type {
  GenerateJob,
  SafeSynthesizerSummary,
} from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { GenerateJobReportRoute } from '@studio/routes/SafeSynthesizerJobReportRoute';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router';

// Mock dependencies
vi.mock('@studio/hooks/useWorkspaceFromPath');
vi.mock('@studio/providers/breadcrumbs/useBreadcrumbs');
vi.mock('@studio/routes/utils');

// Mock navigation component
vi.mock('@studio/components/SafeSynthesizerNavigation', () => ({
  SafeSynthesizerNavigation: ({ selected, jobName }: { selected: string; jobName: string }) => (
    <div data-testid="safe-synthesizer-navigation">
      <span data-testid="navigation-selected">{selected}</span>
      <span data-testid="navigation-job-id">{jobName}</span>
    </div>
  ),
}));

// Mock panel components
vi.mock('@studio/routes/SafeSynthesizerJobReportRoute/components/OverviewPanel', () => ({
  OverviewPanel: ({
    jobId,
    title,
    icon,
  }: {
    jobId: string;
    title: string;
    icon: React.ReactNode;
  }) => (
    <div data-testid="overview-panel">
      <span data-testid="panel-title">{title}</span>
      <span data-testid="panel-job-id">{jobId}</span>
      {icon}
    </div>
  ),
}));

vi.mock(
  '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/SyntheticQualityPanel',
  () => ({
    SyntheticQualityPanel: ({
      reportSummary,
      title,
      icon,
    }: {
      reportSummary?: SafeSynthesizerSummary;
      title: string;
      icon: React.ReactNode;
    }) => (
      <div data-testid="synthetic-quality-panel">
        <span data-testid="panel-title">{title}</span>
        <span data-testid="panel-summary">{reportSummary ? 'has-summary' : 'no-summary'}</span>
        {icon}
      </div>
    ),
  })
);

vi.mock(
  '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/DataPrivacyPanel',
  () => ({
    DataPrivacyPanel: ({
      reportSummary,
      dpEnabled,
      title,
      icon,
    }: {
      reportSummary?: SafeSynthesizerSummary;
      dpEnabled: boolean;
      title: string;
      icon: React.ReactNode;
    }) => (
      <div data-testid="data-privacy-panel">
        <span data-testid="panel-title">{title}</span>
        <span data-testid="panel-dp-enabled">{dpEnabled ? 'enabled' : 'disabled'}</span>
        <span data-testid="panel-summary">{reportSummary ? 'has-summary' : 'no-summary'}</span>
        {icon}
      </div>
    ),
  })
);

// Mock ReportMenu component
vi.mock('@studio/routes/SafeSynthesizerJobReportRoute/components/ReportMenu', () => ({
  ReportMenu: ({
    items,
    onSectionChange,
  }: {
    items: Array<{ id: string; label: string }>;
    onSectionChange: (section: string) => void;
  }) => (
    <div data-testid="report-menu">
      {items.map((item) => (
        <button
          key={item.id}
          data-testid={`menu-item-${item.id}`}
          onClick={() => onSectionChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  ),
}));

// Mock icons - use importOriginal to preserve other exports
vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

const mockuseWorkspaceFromPath = vi.mocked(useWorkspaceFromPath);
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
            initialEntries={['/projects/test-project/safe-synthesizer/jobs/test-job-123/report']}
          >
            <Routes>
              <Route
                path="/projects/:project/safe-synthesizer/jobs/:safeSynthesizerJobName/report"
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
    status: 'COMPLETED',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T01:00:00Z',
    spec: {
      config: {
        data_source: {
          type: 'file',
          path: '/path/to/data.csv',
        },
        privacy: {
          dp_enabled: true,
        },
      },
    },
    ...overrides,
  }) as GenerateJob;

// Helper function to create mock summary
const createMockSummary = (overrides?: Partial<SafeSynthesizerSummary>): SafeSynthesizerSummary =>
  ({
    synthetic_data_quality_score: 8.5,
    data_privacy_score: 7.2,
    ...overrides,
  }) as SafeSynthesizerSummary;

// Helper function to mock API hooks
const mockApiHooks = (job: GenerateJob, summary: SafeSynthesizerSummary) => {
  vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerGetJobSuspense').mockImplementation(
    vi.fn().mockReturnValue({ data: job })
  );
  vi.spyOn(safeSynthesizerApi, 'useSafeSynthesizerDownloadJobResultSummary').mockImplementation(
    vi.fn().mockReturnValue({ data: summary })
  );
};

describe('GenerateJobReportRoute - Feature Flag', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('should be defined when feature flag is enabled', async () => {
    // Mock the environment constant to enable the component
    vi.doMock('@studio/constants/environment', () => ({
      SAFE_SYNTHESIZER_ENABLED: true,
    }));

    const module = await import('./index');
    expect(module.GenerateJobReportRoute).toBeDefined();
    expect(module.GenerateJobReportRoute).not.toBeNull();
  });

  it('should be null when feature flag is disabled', async () => {
    // Mock the environment constant to disable the component
    vi.doMock('@studio/constants/environment', () => ({
      SAFE_SYNTHESIZER_ENABLED: false,
    }));

    const module = await import('./index');
    expect(module.GenerateJobReportRoute).toBeNull();
  });
});

describe('GenerateJobReportRoute - Rendering', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockuseWorkspaceFromPath.mockReturnValue('test-workspace');
    mockUseBreadcrumbs.mockReturnValue({
      breadcrumbs: [],
      setBreadcrumbs: vi.fn(),
    });
  });

  it('should render the route with all panels', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    // Verify navigation is rendered
    expect(screen.getByTestId('safe-synthesizer-navigation')).toBeInTheDocument();
    expect(screen.getByTestId('navigation-selected')).toHaveTextContent('report');
    expect(screen.getByTestId('navigation-job-id')).toHaveTextContent('test-job-123');

    // Verify all three panels are rendered
    expect(screen.getByTestId('overview-panel')).toBeInTheDocument();
    expect(screen.getByTestId('synthetic-quality-panel')).toBeInTheDocument();
    expect(screen.getByTestId('data-privacy-panel')).toBeInTheDocument();
  });

  it('should render menu with correct items', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    // Verify report menu is rendered
    expect(screen.getByTestId('report-menu')).toBeInTheDocument();

    // Verify all menu items are present
    expect(screen.getByTestId('menu-item-overview')).toBeInTheDocument();
    expect(screen.getByTestId('menu-item-synthetic-quality')).toBeInTheDocument();
    expect(screen.getByTestId('menu-item-data-privacy')).toBeInTheDocument();

    // Verify menu item labels
    expect(screen.getByTestId('menu-item-overview')).toHaveTextContent('Overview');
    expect(screen.getByTestId('menu-item-synthetic-quality')).toHaveTextContent(
      'Synthetic Quality'
    );
    expect(screen.getByTestId('menu-item-data-privacy')).toHaveTextContent('Data Privacy');
  });

  it('should render panels with correct titles and icons', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    // Verify panel titles
    const panelTitles = screen.getAllByTestId('panel-title');
    expect(panelTitles).toHaveLength(3);
    expect(panelTitles[0]).toHaveTextContent('Overview');
    expect(panelTitles[1]).toHaveTextContent('Synthetic Quality');
    expect(panelTitles[2]).toHaveTextContent('Data Privacy');

    // Verify icons are rendered
    expect(screen.getByTestId('file-icon')).toBeInTheDocument();
    expect(screen.getByTestId('badge-check-icon')).toBeInTheDocument();
    expect(screen.getByTestId('lock-icon')).toBeInTheDocument();
  });

  it('should pass correct props to OverviewPanel', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    const overviewPanel = screen.getByTestId('overview-panel');
    expect(within(overviewPanel).getByTestId('panel-job-id')).toHaveTextContent('test-job-123');
  });

  it('should pass report summary to score panels', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    // Verify summary is passed to both score panels
    const syntheticQualityPanel = screen.getByTestId('synthetic-quality-panel');
    const dataPrivacyPanel = screen.getByTestId('data-privacy-panel');

    expect(within(syntheticQualityPanel).getByTestId('panel-summary')).toHaveTextContent(
      'has-summary'
    );
    expect(within(dataPrivacyPanel).getByTestId('panel-summary')).toHaveTextContent('has-summary');
  });

  it('should pass dpEnabled flag to DataPrivacyPanel based on job config', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({
      spec: {
        data_source: 'hf://test-namespace/test-dataset/train.jsonl',
        config: {
          privacy: {
            dp_enabled: true,
          },
        },
      },
    } as Partial<GenerateJob>);
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    const dataPrivacyPanel = screen.getByTestId('data-privacy-panel');
    expect(within(dataPrivacyPanel).getByTestId('panel-dp-enabled')).toHaveTextContent('enabled');
  });

  it('should default dpEnabled to false when not specified in job config', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob({
      spec: {
        data_source: 'hf://test-namespace/test-dataset/train.jsonl',
        config: {},
      },
    } as Partial<GenerateJob>);
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    const dataPrivacyPanel = screen.getByTestId('data-privacy-panel');
    expect(within(dataPrivacyPanel).getByTestId('panel-dp-enabled')).toHaveTextContent('disabled');
  });

  it('should set up breadcrumbs correctly', () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    // Verify breadcrumbs were called with correct items
    expect(mockUseBreadcrumbs).toHaveBeenCalledWith(
      expect.objectContaining({
        items: expect.arrayContaining([
          expect.objectContaining({
            slotLabel: 'Safe Synthesizer',
          }),
          expect.objectContaining({
            slotLabel: 'Job Report',
          }),
        ]),
      })
    );
  });

  it('should handle menu section navigation', async () => {
    if (!GenerateJobReportRoute) {
      // Skip test if feature flag is disabled
      return;
    }

    const mockJob = createMockJob();
    const mockSummary = createMockSummary();
    mockApiHooks(mockJob, mockSummary);

    // Mock scrollIntoView
    const scrollIntoViewMock = vi.fn();
    window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;

    render(<GenerateJobReportRoute />, { wrapper: createWrapper() });

    const user = userEvent.setup();

    // Click on synthetic quality menu item
    await user.click(screen.getByTestId('menu-item-synthetic-quality'));

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalledWith({
        behavior: 'smooth',
        block: 'start',
      });
    });
  });
});
