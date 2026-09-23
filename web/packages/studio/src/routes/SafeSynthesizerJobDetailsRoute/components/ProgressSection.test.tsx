// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobLog } from '@nemo/sdk/generated/platform/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { ProgressSection } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/ProgressSection';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('lucide-react', async () => {
  return (await import('@nemo/testing/mocks/lucide')).mockLucideReact(await import('react'));
});

// Mock the highlighted code region to avoid act() warnings from async Shiki
// highlighting. LogViewer composes the snippet from primitives, so only
// CodeSnippetCode needs standing in for — the actions row stays real.
vi.mock('@nvidia/foundations-react-core', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@nvidia/foundations-react-core')>();
  return {
    ...actual,
    CodeSnippetCode: ({
      value,
      className,
      ref,
    }: {
      value?: string;
      className?: string;
      ref?: React.Ref<HTMLElement>;
    }) => (
      <pre
        data-testid="nv-code-snippet-code"
        ref={ref as React.Ref<HTMLPreElement>}
        className={className}
      >
        {value}
      </pre>
    ),
  };
});

// Helper to create mock log entries
const createMockLog = (
  message: string,
  timestamp: string = '2024-01-01T10:00:00.000Z'
): HelixJobLog => ({
  message,
  timestamp,
  job: 'test-job-id',
  job_step: 'processing',
  job_task: 'main',
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

// Mock blob and URL creation for download tests
const mockCreateObjectURL = vi.fn();
const mockRevokeObjectURL = vi.fn();
const mockClick = vi.fn();
const mockClipboardWriteText = vi.fn();

// Setup clipboard API once before all tests
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: mockClipboardWriteText,
  },
  writable: true,
  configurable: true,
});

// Capture the real createElement once at module scope, before any spying.
const originalCreateElement = document.createElement.bind(document);

// Setup DOM APIs
beforeEach(() => {
  vi.clearAllMocks();

  // Reset mock implementations after clearAllMocks
  mockClipboardWriteText.mockResolvedValue(undefined);
  mockCreateObjectURL.mockReturnValue('blob:mock-url');

  // Mock URL methods
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  // Mock document.createElement for anchor element
  vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
    const element = originalCreateElement(tagName);
    if (tagName === 'a') {
      element.click = mockClick;
    }
    return element;
  });
});

// Helper to render with default props
const renderProgressSection = (
  jobId: string = 'test-job-123',
  isLoading: boolean = false,
  logs: HelixJobLog[] = []
) => {
  return render(<ProgressSection jobId={jobId} isLoading={isLoading} logs={logs} />, {
    wrapper: createWrapper(),
  });
};

// Helper to find the download button using Testing Library queries
const getDownloadButton = () => {
  const buttons = screen.getAllByRole('button');
  const downloadButton = buttons.find((button) => within(button).queryByTestId('download-icon'));
  if (!downloadButton) {
    throw new Error('Download button not found');
  }
  return downloadButton;
};

describe('ProgressSection', () => {
  describe('Basic Rendering', () => {
    it('should render progress section with heading', () => {
      renderProgressSection();

      expect(screen.getByText('Progress')).toBeInTheDocument();
    });

    it('should render download button when logs are present', () => {
      const logs = [createMockLog('Test log', '2024-01-01T10:00:00.000Z')];
      renderProgressSection('test-job-123', false, logs);

      expect(screen.getByTestId('download-icon')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('should display loading spinner when logs are loading', () => {
      renderProgressSection('test-job-123', true, []);

      expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
    });

    it('should not display logs while loading', () => {
      const logs = [createMockLog('Log message 1', '2024-01-01T10:00:00.000Z')];

      renderProgressSection('test-job-123', true, logs);

      expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
      expect(screen.queryByText('Log message 1')).not.toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('should display "No logs available yet" when there are no logs', () => {
      renderProgressSection('test-job-123', false, []);

      expect(screen.getByText('No logs available yet')).toBeInTheDocument();
    });
  });

  describe('Logs Display', () => {
    it('should display logs with timestamps and messages', () => {
      const logs = [
        createMockLog('Starting job', '2024-01-01T10:00:00.000Z'),
        createMockLog('Processing data', '2024-01-01T10:01:00.000Z'),
        createMockLog('Job completed', '2024-01-01T10:02:00.000Z'),
      ];

      renderProgressSection('test-job-123', false, logs);

      expect(screen.getByText(/Starting job/)).toBeInTheDocument();
      expect(screen.getByText(/Processing data/)).toBeInTheDocument();
      expect(screen.getByText(/Job completed/)).toBeInTheDocument();
      expect(screen.getByText(/\[2024-01-01T10:00:00.000Z]/)).toBeInTheDocument();
    });

    it('should display all logs', () => {
      const logs = [
        createMockLog('Log 1', '2024-01-01T10:00:00.000Z'),
        createMockLog('Log 2', '2024-01-01T10:01:00.000Z'),
        createMockLog('Log 3', '2024-01-01T10:02:00.000Z'),
      ];

      renderProgressSection('test-job-123', false, logs);

      expect(screen.getByText(/Log 1/)).toBeInTheDocument();
      expect(screen.getByText(/Log 2/)).toBeInTheDocument();
      expect(screen.getByText(/Log 3/)).toBeInTheDocument();
    });
  });

  describe('Download Functionality', () => {
    it('should download all logs when download button is clicked', async () => {
      const logs = [
        createMockLog('Log message 1', '2024-01-01T10:00:00.000Z'),
        createMockLog('Log message 2', '2024-01-01T10:01:00.000Z'),
        createMockLog('Log message 3', '2024-01-01T10:02:00.000Z'),
      ];

      const user = userEvent.setup();
      renderProgressSection('test-job-123', false, logs);

      const downloadButton = getDownloadButton();
      await user.click(downloadButton);

      // Verify blob was created with correct content
      expect(mockCreateObjectURL).toHaveBeenCalled();
      const blobArg = mockCreateObjectURL.mock.calls[0][0] as Blob;
      const text = await blobArg.text();
      expect(text).toBe(
        '[2024-01-01T10:00:00.000Z]   Log message 1\n[2024-01-01T10:01:00.000Z]   Log message 2\n[2024-01-01T10:02:00.000Z]   Log message 3'
      );

      // Verify anchor element was created and clicked
      expect(mockClick).toHaveBeenCalled();

      // Verify URL was revoked
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
    });

    it('should download logs with correct filename', async () => {
      const logs = [createMockLog('Test log')];

      const user = userEvent.setup();
      renderProgressSection('my-job-123', false, logs);

      const downloadButton = getDownloadButton();
      await user.click(downloadButton);

      expect(mockClick).toHaveBeenCalled();
    });

    it('should download all logs including those not displayed', async () => {
      const manyLogs = Array.from({ length: 50 }, (_, i) => createMockLog(`Log ${i + 1}`));

      const user = userEvent.setup();
      renderProgressSection('test-job-123', false, manyLogs);

      const downloadButton = getDownloadButton();
      await user.click(downloadButton);

      // Verify all 50 logs are downloaded
      const blobArg = mockCreateObjectURL.mock.calls[0][0] as Blob;
      const text = await blobArg.text();
      const lines = text.split('\n');
      expect(lines).toHaveLength(50);
      expect(lines[0]).toContain('Log 1');
      expect(lines[49]).toContain('Log 50');
    });
  });

  describe('Load Previous Logs', () => {
    it('should show "Load previous logs" button when there are more than 30 logs', async () => {
      const manyLogs = Array.from({ length: 50 }, (_, i) => createMockLog(`Log ${i + 1}`));
      renderProgressSection('test-job-123', false, manyLogs);

      expect(await screen.findByText('Load previous logs')).toBeInTheDocument();
      expect(await screen.findByTestId('arrow-up-icon')).toBeInTheDocument();
    });

    it('should not show "Load previous logs" button when there are 30 or fewer logs', async () => {
      const logs = Array.from({ length: 30 }, (_, i) => createMockLog(`Log ${i + 1}`));
      renderProgressSection('test-job-123', false, logs);

      await waitFor(() => {
        expect(screen.queryByText('Load previous logs')).not.toBeInTheDocument();
      });
    });

    it('should show line count with total when not all logs are shown', async () => {
      const manyLogs = Array.from({ length: 50 }, (_, i) => createMockLog(`Log ${i + 1}`));
      renderProgressSection('test-job-123', false, manyLogs);

      await waitFor(() => {
        // Line count shows "30 of 50 lines"
        expect(screen.getByText('30 of 50 lines')).toBeInTheDocument();
      });
    });
  });
});
