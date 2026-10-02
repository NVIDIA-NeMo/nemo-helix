// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MockToastProvider } from '@nemo/common/src/tests/MockToastProvider';
import { useFilesDownloadFile } from '@nemo/sdk/generated/platform/files';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import {
  useSafeSynthesizerDownloadJobResultSyntheticData,
  useSafeSynthesizerListJobResults,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { type GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { SafeSynthesizerFilesetPreview } from '@studio/components/SafeSynthesizerFilesetPreview';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Papa from 'papaparse';
import { MemoryRouter } from 'react-router';

// Mock Papa.parse
vi.mock('papaparse', () => ({
  default: {
    parse: vi.fn(),
  },
}));

// Mock generated SafeSynthesizer API hooks
vi.mock('@nemo/sdk/generated/safe-synthesizer/safe-synthesizer', () => ({
  useSafeSynthesizerListJobResults: vi.fn(),
  useSafeSynthesizerDownloadJobResultSyntheticData: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/platform/files', () => ({
  useFilesDownloadFile: vi.fn(),
}));

// Mock child components
vi.mock('@studio/components/SafeSynthesizerFilesetPreview/FilePreview', () => ({
  FilePreview: ({
    title,
    onClose,
    isLoading,
    error,
    children,
  }: {
    title: string;
    onClose: () => void;
    isLoading: boolean;
    error?: string;
    children?: React.ReactNode;
  }) => (
    <div data-testid="file-preview">
      <div data-testid="file-preview-title">{title}</div>
      {isLoading && <div data-testid="file-preview-loading">Loading...</div>}
      {error && <div data-testid="file-preview-error">{error}</div>}
      {!isLoading && !error && children}
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

vi.mock('@nemo/common/src/components/ScrollTable', () => ({
  ScrollTable: ({ rows, columns }: { rows: unknown[]; columns: unknown[] }) => (
    <div data-testid="scroll-table">
      <div data-testid="scroll-table-rows">{JSON.stringify(rows)}</div>
      <div data-testid="scroll-table-columns">{JSON.stringify(columns)}</div>
    </div>
  ),
}));

vi.mock('@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobConfigDrawer', () => ({
  JobConfigDrawer: ({ open }: { open: boolean }) =>
    open ? <div data-testid="job-config-drawer">Job Config Drawer</div> : null,
}));

vi.mock('@studio/components/SafeSynthesizerFilesetPreview/FilesetFilePreviewLink', () => ({
  FilesetFilePreviewLink: ({ url, children }: { url: string; children?: React.ReactNode }) => (
    <div data-testid="fileset-preview-link" data-url={url}>
      {children}
    </div>
  ),
}));

// Mock parseFilesetUrl utility
vi.mock('@nemo/common/src/components/DatasetFileSelect/utils', () => ({
  parseFilesetUrl: vi.fn((url: string) => {
    if (!url || !url.startsWith('fileset://')) {
      return null;
    }
    // Parse fileset://workspace/name/path
    const withoutProtocol = url.replace('fileset://', '');
    const parts = withoutProtocol.split('/');
    return {
      workspace: parts[0],
      name: parts[1],
      path: parts.slice(2).join('/'),
    };
  }),
}));

const mockUseListJobResults = vi.mocked(useSafeSynthesizerListJobResults);
const mockUseDownloadJobResultSyntheticData = vi.mocked(
  useSafeSynthesizerDownloadJobResultSyntheticData
);
const mockUseDownloadFilesetsFileV2 = vi.mocked(useFilesDownloadFile);
const mockPapaParse = vi.mocked(Papa.parse);

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
        <MockToastProvider>
          <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        </MockToastProvider>
      </ThemeProvider>
    </MemoryRouter>
  );
};

// Mock data - all jobs now use fileset:// URLs
const createMockJob = (overrides?: Partial<GenerateJob>): GenerateJob => ({
  id: 'test-job-id',
  name: 'test-job',
  workspace: 'test-workspace',
  status: HelixJobStatus.completed,
  spec: {
    data_source: 'fileset://test-workspace/test-dataset/source-data.csv',
    config: {},
    ...overrides?.spec,
  },
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  ...overrides,
});

const createMockJobWithFilesetUrl = (overrides?: Partial<GenerateJob>): GenerateJob => ({
  id: 'test-job-id',
  name: 'test-job',
  workspace: 'test-workspace',
  status: HelixJobStatus.completed,
  spec: {
    data_source: 'fileset://test-workspace/test-dataset/source-data.csv',
    config: {},
    ...overrides?.spec,
  },
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  ...overrides,
});

describe('SafeSynthesizerFilesetPreview', () => {
  beforeEach(() => {
    mockUseParams({
      workspace: 'test-workspace',
      safeSynthesizerJobName: 'test-job-id',
    });

    mockUseListJobResults.mockReturnValue({
      data: {
        data: [
          { artifact_url: 'url1', artifact_type: 'type1' },
          { artifact_url: 'url2', artifact_type: 'type2' },
          {
            artifact_url: 'fileset://test-workspace/test-dataset/synthetic-data.csv',
            artifact_type: 'synthetic_data',
          },
        ],
      },
      isLoading: false,
      error: null,
    } as never);

    mockUseDownloadJobResultSyntheticData.mockReturnValue({
      refetch: vi.fn(),
      data: undefined,
      isLoading: false,
      error: null,
    } as never);

    mockUseDownloadFilesetsFileV2.mockReturnValue({
      refetch: vi.fn().mockResolvedValue({
        data: new Blob(['test file content'], { type: 'text/plain' }),
      }),
      data: undefined,
      isLoading: false,
      error: null,
    } as never);

    mockPapaParse.mockReturnValue({
      data: [
        { id: '1', name: 'Test 1' },
        { id: '2', name: 'Test 2' },
      ],
      meta: { fields: ['id', 'name'] },
      errors: [],
    } as never);
  });

  describe('Basic Rendering', () => {
    it('should render data source key-value', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Data Source')).toBeInTheDocument();
      expect(screen.getByText('test-workspace/test-dataset/source-data.csv')).toBeInTheDocument();
    });

    it('should render generation results key-value', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });
      expect(screen.getByText('Generation Results')).toBeInTheDocument();
      expect(screen.getByText('synthetic-data.csv')).toBeInTheDocument();
    });

    it('should not render file preview by default', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(screen.queryByTestId('file-preview')).not.toBeInTheDocument();
    });
  });

  describe('API Hook Configuration', () => {
    it('should enable job results query when job status is completed', () => {
      const job = createMockJob({ status: HelixJobStatus.completed });
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(mockUseListJobResults).toHaveBeenCalledWith(
        'test-workspace',
        'test-job-id',
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: true,
          }),
        })
      );
    });

    it('should disable job results query when job status is not completed', () => {
      const job = createMockJob({ status: HelixJobStatus.active });
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(mockUseListJobResults).toHaveBeenCalledWith(
        'test-workspace',
        'test-job-id',
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: false,
          }),
        })
      );
    });

    it('should disable results query when job is not terminated', () => {
      const job = createMockJob({ status: HelixJobStatus.active });
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(mockUseListJobResults).toHaveBeenCalledWith(
        'test-workspace',
        'test-job-id',
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: false,
          }),
        })
      );
    });

    it('should enable results query when job is terminated', () => {
      const job = createMockJob({ status: HelixJobStatus.completed });
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(mockUseListJobResults).toHaveBeenCalledWith(
        'test-workspace',
        'test-job-id',
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: true,
          }),
        })
      );
    });

    it('should disable synthetic data query by default', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(mockUseDownloadJobResultSyntheticData).toHaveBeenCalledWith(
        'test-workspace',
        'test-job-id',
        expect.objectContaining({
          query: expect.objectContaining({
            enabled: false,
          }),
        })
      );
    });
  });

  // Note: Data Source Preview functionality is now handled by FilesetFilePreviewLink
  // which is tested separately in FilesetFilePreviewLink.test.tsx
  describe('Data Source Preview', () => {
    it('should render FilesetFilePreviewLink for data source', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const previewLink = screen.getByTestId('fileset-preview-link');
      expect(previewLink).toBeInTheDocument();
      expect(previewLink).toHaveAttribute('data-url', job.spec.data_source);
    });
  });

  describe('Synthetic Data Preview', () => {
    it('should show file preview when generation results link is clicked', async () => {
      const user = userEvent.setup();
      const mockBlob = new Blob(['id,name\n1,Test 1\n2,Test 2'], { type: 'text/csv' });
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview')).toBeInTheDocument();
      });
    });

    it('should parse CSV data when fetching synthetic data', async () => {
      const user = userEvent.setup();
      const csvContent = 'id,name\n1,Test 1\n2,Test 2';
      const mockBlob = {
        text: vi.fn().mockResolvedValue(csvContent),
      };
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });

      // Wait for Papa.parse to be called with the CSV content
      await waitFor(
        () => {
          expect(mockPapaParse).toHaveBeenCalledWith(csvContent, { header: true });
        },
        { timeout: 3000 }
      );
    });

    it('should show error when synthetic data parsing fails', async () => {
      const user = userEvent.setup();
      const mockBlob = {
        text: vi.fn().mockRejectedValue(new Error('Parse failed')),
      };
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview-error')).toHaveTextContent(
          'Could not parse synthetic data'
        );
      });
    });

    it('should handle empty response from synthetic data fetch', async () => {
      const user = userEvent.setup();
      const mockRefetch = vi.fn().mockResolvedValue({
        data: undefined,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    it('should display parsed synthetic data in file preview', async () => {
      const user = userEvent.setup();
      const csvContent = 'id,name\n1,Test 1\n2,Test 2';
      const mockBlob = {
        text: vi.fn().mockResolvedValue(csvContent),
      };
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        // The component currently uses sourceUrl for the title regardless of preview type
        expect(screen.getByTestId('file-preview')).toBeInTheDocument();
      });

      // Verify ScrollTable is rendered with synthetic data
      await waitFor(() => {
        expect(screen.getByTestId('scroll-table')).toBeInTheDocument();
      });
    });
  });

  describe('File Preview Panel', () => {
    it('should close file preview when close button is clicked', async () => {
      const user = userEvent.setup();
      const csvContent = 'id,name\n1,Test 1\n2,Test 2';
      const mockBlob = {
        text: vi.fn().mockResolvedValue(csvContent),
      };
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      // Open preview
      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview')).toBeInTheDocument();
      });

      // Close preview
      const closeButton = screen.getByText('Close');
      await user.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('file-preview')).not.toBeInTheDocument();
      });
    });

    it('should show loading state in file preview', async () => {
      const user = userEvent.setup();
      let resolvePromise: (value: { data: Blob }) => void;
      const mockRefetch = vi.fn().mockReturnValue(
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
      );

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByTestId('file-preview-loading')).toBeInTheDocument();
      });

      // Resolve the promise
      const mockBlob = new Blob(['id,name\n1,Test 1\n2,Test 2'], { type: 'text/csv' });
      resolvePromise!({ data: mockBlob });

      await waitFor(() => {
        expect(screen.queryByTestId('file-preview-loading')).not.toBeInTheDocument();
      });
    });
  });

  describe('Generation Results Display', () => {
    it('should show generation results when job results contain synthetic data', () => {
      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('synthetic-data.csv')).toBeInTheDocument();
    });

    it('should not show generation results when job results are empty', () => {
      mockUseListJobResults.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      // Generation results key-value should still render but without a value
      expect(screen.getByText('Generation Results')).toBeInTheDocument();
      expect(screen.getByText('-')).toBeInTheDocument();
    });

    it('should extract path from synthetic data artifact URL', () => {
      mockUseListJobResults.mockReturnValue({
        data: {
          data: [
            {},
            {},
            {
              artifact_url: 'fileset://namespace/repo/path/to/output.csv',
              artifact_type: 'synthetic_data',
            },
          ],
        },
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      // The component displays the full path from the fileset URL
      expect(screen.getByText('path/to/output.csv')).toBeInTheDocument();
    });
  });

  describe('Loading States', () => {
    it('should set loading state when previewing synthetic data', async () => {
      const user = userEvent.setup();
      let resolvePromise: (value: { data: Blob }) => void;
      const mockRefetch = vi.fn().mockReturnValue(
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
      );

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview-loading')).toBeInTheDocument();
      });

      const mockBlob = new Blob(['id,name\n1,Test 1\n2,Test 2'], { type: 'text/csv' });
      resolvePromise!({ data: mockBlob });

      await waitFor(() => {
        expect(screen.queryByTestId('file-preview-loading')).not.toBeInTheDocument();
      });
    });

    it('should clear loading state after closing panel', async () => {
      const user = userEvent.setup();
      const csvContent = 'id,name\n1,Test 1\n2,Test 2';
      const mockBlob = {
        text: vi.fn().mockResolvedValue(csvContent),
      };
      const mockRefetch = vi.fn().mockResolvedValue({
        data: mockBlob,
      });

      mockUseDownloadJobResultSyntheticData.mockReturnValue({
        refetch: mockRefetch,
        data: undefined,
        isLoading: false,
        error: null,
      } as never);

      const job = createMockJob();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      // Open preview
      const syntheticDataLink = screen.getByText('synthetic-data.csv');
      await user.click(syntheticDataLink);

      await waitFor(() => {
        expect(screen.getByTestId('file-preview')).toBeInTheDocument();
      });

      // Close preview
      const closeButton = screen.getByText('Close');
      await user.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('file-preview')).not.toBeInTheDocument();
      });
    });
  });

  describe('Fileset URL Support', () => {
    it('should display fileset data source correctly', () => {
      const job = createMockJobWithFilesetUrl();
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('Data Source')).toBeInTheDocument();
      expect(screen.getByText('test-workspace/test-dataset/source-data.csv')).toBeInTheDocument();
    });

    it('should parse nested fileset paths', () => {
      const job = createMockJobWithFilesetUrl({
        spec: {
          data_source: 'fileset://workspace/dataset/folder/subfolder/file.jsonl',
          config: {},
        },
      });
      render(<SafeSynthesizerFilesetPreview job={job} />, { wrapper: createWrapper() });

      expect(screen.getByText('workspace/dataset/folder/subfolder/file.jsonl')).toBeInTheDocument();
    });
  });
});
