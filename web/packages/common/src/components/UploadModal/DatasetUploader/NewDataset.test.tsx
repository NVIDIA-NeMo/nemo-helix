// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type UploadModalContextType,
  UploadModalContext,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import {
  type UploadModalState,
  initialState,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { NewDataset } from '@nemo/common/src/components/UploadModal/DatasetUploader/NewDataset';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';

vi.mock('@nemo/common/src/components/UploadModal/FileUpload', () => ({
  FileUpload: ({ error }: { error?: string }) => (
    <div data-testid="file-upload">
      <button>Choose files</button>
      {error && <div data-testid="file-upload-error">{error}</div>}
    </div>
  ),
}));

vi.mock('@nemo/common/src/components/UploadModal/SimpleFilesTable', () => ({
  SimpleFilesTable: () => <div data-testid="simple-files-table">Files Table</div>,
}));

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
    ...initialState,
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

describe('NewDataset', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders dataset name input', () => {
      render(<NewDataset />, { wrapper: createWrapper() });

      expect(screen.getByText('Dataset Name')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Name this Dataset')).toBeInTheDocument();
    });

    it('renders file upload when no files are present', () => {
      render(<NewDataset />, { wrapper: createWrapper() });

      expect(screen.getByText('File')).toBeInTheDocument();
      expect(screen.getByTestId('file-upload')).toBeInTheDocument();
    });

    it('renders SimpleFilesTable when files are present', () => {
      render(<NewDataset />, {
        wrapper: createWrapper({
          files: [
            {
              id: 'file1',
              type: 'new',
              file: new File(['content'], 'test.json', { type: 'application/json' }),
            },
          ],
        }),
      });

      expect(screen.getByTestId('simple-files-table')).toBeInTheDocument();
      expect(screen.queryByTestId('file-upload')).not.toBeInTheDocument();
    });

    it('returns null when dataset type is existing', () => {
      const { container } = render(<NewDataset />, {
        wrapper: createWrapper({
          dataset: {
            type: 'existing',
            dataset: {
              id: '123',
              name: 'Existing Dataset',
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
        }),
      });

      expect(container).toBeEmptyDOMElement();
    });

    it('displays dataset name from state', () => {
      render(<NewDataset />, {
        wrapper: createWrapper({
          dataset: {
            type: 'new',
            name: 'My Dataset',
          },
        }),
      });

      const input = screen.getByPlaceholderText('Name this Dataset');
      expect(input).toHaveValue('My Dataset');
    });
  });

  describe('error handling', () => {
    it('displays dataset name error when present', () => {
      render(<NewDataset />, {
        wrapper: createWrapper({
          errors: {
            datasetName: 'Dataset name is required',
          },
        }),
      });

      expect(screen.getByText('Dataset name is required')).toBeInTheDocument();
    });

    it('passes file error to FileUpload component', () => {
      render(<NewDataset />, {
        wrapper: createWrapper({
          errors: {
            file: 'File is required',
          },
        }),
      });

      expect(screen.getByTestId('file-upload-error')).toHaveTextContent('File is required');
    });

    it('does not show error when errors object is empty', () => {
      render(<NewDataset />, {
        wrapper: createWrapper({
          errors: {},
        }),
      });

      expect(screen.queryByText(/is required/i)).not.toBeInTheDocument();
    });
  });

  describe('user interactions', () => {
    it('dispatches UPDATE_DATASET action when dataset name changes', async () => {
      const mockDispatch = vi.fn();
      render(<NewDataset />, { wrapper: createWrapper({}, mockDispatch) });

      const input = screen.getByPlaceholderText('Name this Dataset');
      await user.type(input, 'my-dataset');

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'UPDATE_DATASET',
        payload: {
          type: 'new',
          name: expect.stringContaining('m'),
        },
      });
    });

    it('updates dataset name value as user types', async () => {
      const mockDispatch = vi.fn();
      render(<NewDataset />, { wrapper: createWrapper({}, mockDispatch) });

      const input = screen.getByPlaceholderText('Name this Dataset');
      await user.type(input, 'test');

      // Should dispatch for each character typed (since state is mocked and doesn't update,
      // each character is typed individually rather than accumulating)
      expect(mockDispatch).toHaveBeenCalledTimes(4);
      expect(mockDispatch).toHaveBeenNthCalledWith(1, {
        type: 'UPDATE_DATASET',
        payload: { type: 'new', name: 't' },
      });
      expect(mockDispatch).toHaveBeenNthCalledWith(2, {
        type: 'UPDATE_DATASET',
        payload: { type: 'new', name: 'e' },
      });
      expect(mockDispatch).toHaveBeenNthCalledWith(3, {
        type: 'UPDATE_DATASET',
        payload: { type: 'new', name: 's' },
      });
      expect(mockDispatch).toHaveBeenNthCalledWith(4, {
        type: 'UPDATE_DATASET',
        payload: { type: 'new', name: 't' },
      });
    });
  });
});
