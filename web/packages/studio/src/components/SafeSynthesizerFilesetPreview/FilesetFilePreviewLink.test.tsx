// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import * as platformApi from '@nemo/sdk/generated/platform/files';
import { ThemeProvider } from '@nvidia/foundations-react-core';
import { FilesetFilePreviewLink } from '@studio/components/SafeSynthesizerFilesetPreview/FilesetFilePreviewLink';
import * as utils from '@studio/components/SafeSynthesizerFilesetPreview/util';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the SDK API hook
vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => {
  const original = await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>();
  return {
    ...original,
    useFilesDownloadFile: vi.fn(),
  };
});

describe('FilesetFilePreviewLink', () => {
  const mockRefetch = vi.fn();
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  const renderComponent = (props: { url: string; children?: React.ReactNode }) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <ToastProvider>
            <FilesetFilePreviewLink {...props} />
          </ToastProvider>
        </ThemeProvider>
      </QueryClientProvider>
    );
  };

  // Helper to create a mock Blob with text() method
  const createMockBlob = (content: string) => new Blob([content], { type: 'text/plain' });

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();

    // Default mock for SDK hook
    vi.mocked(platformApi.useFilesDownloadFile).mockReturnValue({
      refetch: mockRefetch,
      data: undefined,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof platformApi.useFilesDownloadFile>);
  });

  describe('rendering', () => {
    it('should render anchor with children', () => {
      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Click to preview</span>,
      });

      expect(screen.getByText('Click to preview')).toBeInTheDocument();
    });

    it('should not show preview panel initially', () => {
      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  describe('CSV file preview', () => {
    it('should fetch and display CSV content when clicked', async () => {
      const user = userEvent.setup();
      const mockCsvData = 'name,age\nJohn,30\nJane,25';

      mockRefetch.mockResolvedValue({
        data: createMockBlob(mockCsvData),
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'csv',
        tabularData: {
          columns: [{ children: 'name' }, { children: 'age' }],
          rows: [
            {
              id: '0',
              cells: [{ children: 'John' }, { children: '30' }],
            },
            {
              id: '1',
              cells: [{ children: 'Jane' }, { children: '25' }],
            },
          ],
        },
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview CSV</span>,
      });

      await user.click(screen.getByText('Preview CSV'));

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(screen.getByText('myorg/mydataset/data/train.csv')).toBeInTheDocument();
      });

      expect(utils.parseFileContent).toHaveBeenCalledWith('data/train.csv', mockCsvData);
    });
  });

  describe('JSON file preview', () => {
    it('should fetch and display JSON content when clicked', async () => {
      const user = userEvent.setup();
      const mockJsonData = '{"name": "John", "age": 30}';

      mockRefetch.mockResolvedValue({
        data: createMockBlob(mockJsonData),
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'json',
        jsonData: mockJsonData,
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/data.json',
        children: <span>Preview JSON</span>,
      });

      await user.click(screen.getByText('Preview JSON'));

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(screen.getByText('myorg/mydataset/data/data.json')).toBeInTheDocument();
      });

      expect(utils.parseFileContent).toHaveBeenCalledWith('data/data.json', mockJsonData);
    });
  });

  describe('error handling', () => {
    it('should display error when fetch fails', async () => {
      const user = userEvent.setup();

      mockRefetch.mockResolvedValue({
        data: null,
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(screen.getByText('Error fetching file')).toBeInTheDocument();
      });
    });

    it('should display error when parsing fails', async () => {
      const user = userEvent.setup();

      mockRefetch.mockRejectedValue(new Error('Network error'));

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(screen.getByText('Could not parse file')).toBeInTheDocument();
      });
    });

    it('should display error for unsupported file types', async () => {
      const user = userEvent.setup();

      mockRefetch.mockResolvedValue({
        data: createMockBlob('some content'),
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'error',
        error: 'Unsupported file type',
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/file.txt',
        children: <span>Preview</span>,
      });

      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(screen.getByText('Unsupported file type')).toBeInTheDocument();
      });
    });
  });

  describe('preview panel interactions', () => {
    it('should close preview panel when close button is clicked', async () => {
      const user = userEvent.setup();
      const mockCsvData = 'name\nJohn';

      mockRefetch.mockResolvedValue({
        data: createMockBlob(mockCsvData),
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'csv',
        tabularData: {
          columns: [{ children: 'name' }],
          rows: [{ id: '0', cells: [{ children: 'John' }] }],
        },
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      // Open preview
      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(screen.getByText('myorg/mydataset/data/train.csv')).toBeInTheDocument();
      });

      // Close preview
      const closeButton = screen.getByRole('button', { name: /Close Side Panel/i });
      await user.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByText('myorg/mydataset/data/train.csv')).not.toBeInTheDocument();
      });
    });

    it('should reset state when opening preview again', async () => {
      const user = userEvent.setup();

      // First click - error
      mockRefetch.mockResolvedValueOnce({
        data: null,
      });

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(screen.getByText('Error fetching file')).toBeInTheDocument();
      });

      // Close
      await user.click(screen.getByRole('button', { name: /Close Side Panel/i }));

      // Second click - success
      mockRefetch.mockResolvedValueOnce({
        data: createMockBlob('name\nJohn'),
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'csv',
        tabularData: {
          columns: [{ children: 'name' }],
          rows: [{ id: '0', cells: [{ children: 'John' }] }],
        },
      });

      await user.click(screen.getByText('Preview'));

      await waitFor(() => {
        expect(screen.getByText('myorg/mydataset/data/train.csv')).toBeInTheDocument();
      });

      expect(screen.queryByText('Error fetching file')).not.toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('should show loading state while fetching', async () => {
      const user = userEvent.setup();

      // Create a promise that we control
      let resolveRefetch!: (value: { data: Blob }) => void;
      const refetchPromise = new Promise<{ data: Blob }>((resolve) => {
        resolveRefetch = resolve;
      });

      mockRefetch.mockReturnValue(refetchPromise);

      renderComponent({
        url: 'fileset://myorg/mydataset/data/train.csv',
        children: <span>Preview</span>,
      });

      await user.click(screen.getByText('Preview'));

      // Should show loading indicator
      await waitFor(() => {
        expect(screen.getByLabelText('Loading...')).toBeInTheDocument();
      });

      vi.spyOn(utils, 'parseFileContent').mockReturnValue({
        type: 'csv',
        tabularData: {
          columns: [{ children: 'name' }],
          rows: [{ id: '0', cells: [{ children: 'John' }] }],
        },
      });

      // Resolve the fetch
      resolveRefetch({
        data: createMockBlob('name\nJohn'),
      });

      await waitFor(() => {
        expect(screen.getByText('John')).toBeInTheDocument();
      });
    });
  });
});
