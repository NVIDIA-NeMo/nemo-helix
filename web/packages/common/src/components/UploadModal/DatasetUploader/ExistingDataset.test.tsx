// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type UploadModalContextType,
  UploadModalContext,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import {
  type UploadModalState,
  uploadModalInitialState,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { ExistingDataset } from '@nemo/common/src/components/UploadModal/DatasetUploader/ExistingDataset';
import type { UploadFile } from '@nemo/common/src/components/UploadModal/types';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';

vi.mock('@nemo/common/src/components/UploadModal/FileUpload', () => ({
  FileUpload: ({ error }: { error?: string }) => (
    <div data-testid="file-upload">
      {error && <div data-testid="file-upload-error">{error}</div>}
    </div>
  ),
}));

vi.mock('@nemo/common/src/components/UploadModal/SimpleFilesTable', () => ({
  SimpleFilesTable: () => <div data-testid="simple-files-table">Files Table</div>,
}));

const mockFiles: UploadFile[] = [
  {
    id: 'existing-1',
    type: 'existing',
    file: {
      path: 'file1.jsonl',
      file_ref: 'ref1',
      size: 1024,
      file_url: 'https://example.com/file1.jsonl',
    },
  },
  {
    id: 'existing-2',
    type: 'existing',
    file: {
      path: 'file2.jsonl',
      file_ref: 'ref2',
      size: 2048,
      file_url: 'https://example.com/file2.jsonl',
    },
  },
];

const createWrapper = (
  stateOverrides?: Partial<UploadModalState>,
  dispatch?: ReturnType<typeof vi.fn>
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  const mockDispatch = dispatch || vi.fn();
  const state: UploadModalState = {
    ...uploadModalInitialState,
    dataset: {
      type: 'existing',
      dataset: {
        id: 'default/test-dataset',
        name: 'test-dataset',
        workspace: 'default',
        description: '',
        purpose: 'dataset',
        storage: { type: 'local', path: '/data' },
        metadata: {},
        custom_fields: {},
        project: 'default',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      },
    },
    ...stateOverrides,
  };

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <UploadModalContext.Provider value={[state, mockDispatch] as UploadModalContextType}>
        {children}
      </UploadModalContext.Provider>
    </QueryClientProvider>
  );
};

describe('ExistingDataset', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('returns null when dataset type is new', () => {
      const { container } = render(<ExistingDataset />, {
        wrapper: createWrapper({
          dataset: { type: 'new', name: 'new-dataset' },
          files: mockFiles,
        }),
      });

      expect(container).toBeEmptyDOMElement();
    });

    it('renders loading spinner while fetching files', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ isFetching: true }),
      });

      expect(screen.getByText('Loading dataset files...')).toBeInTheDocument();
    });

    it('shows file upload UI when dataset has no files and no selected files', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: [], selectedFiles: [], isFetching: false }),
      });

      expect(screen.getByText('File')).toBeInTheDocument();
      expect(screen.getByText('There are no files found in this dataset.')).toBeInTheDocument();
      expect(screen.getByTestId('file-upload')).toBeInTheDocument();
    });

    it('renders SimpleFilesTable when files are present', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: mockFiles, isFetching: false }),
      });

      expect(screen.getByText('Files')).toBeInTheDocument();
      expect(screen.getByTestId('simple-files-table')).toBeInTheDocument();
    });

    it('does not show file upload when files are present', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: mockFiles, isFetching: false }),
      });

      expect(
        screen.queryByText('There are no files found in this dataset.')
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId('file-upload')).not.toBeInTheDocument();
    });
  });

  describe('error handling', () => {
    it('passes file error to FileUpload component', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({
          files: [],
          selectedFiles: [],
          isFetching: false,
          errors: {
            file: 'File is required',
          },
        }),
      });

      expect(screen.getByTestId('file-upload-error')).toHaveTextContent('File is required');
    });

    it('does not show error when errors object is empty', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({
          files: [],
          selectedFiles: [],
          isFetching: false,
          errors: {},
        }),
      });

      expect(screen.queryByTestId('file-upload-error')).not.toBeInTheDocument();
    });
  });

  describe('conditional rendering logic', () => {
    it('shows file upload when files array is empty', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: [], selectedFiles: [], isFetching: false }),
      });

      expect(screen.getByTestId('file-upload')).toBeInTheDocument();
    });

    it('shows SimpleFilesTable when at least one file exists', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: mockFiles, isFetching: false }),
      });

      expect(screen.getByTestId('simple-files-table')).toBeInTheDocument();
    });

    it('does not show loading spinner when isFetching is false', () => {
      render(<ExistingDataset />, {
        wrapper: createWrapper({ files: mockFiles, isFetching: false }),
      });

      expect(screen.queryByText('Loading dataset files...')).not.toBeInTheDocument();
    });
  });
});
