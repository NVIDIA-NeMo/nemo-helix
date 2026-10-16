// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type UploadModalContextType,
  UploadModalContext,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalContext';
import {
  UploadModalState,
  initialState,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { SimpleFilesTable } from '@nemo/common/src/components/UploadModal/SimpleFilesTable';
import { UploadFile } from '@nemo/common/src/components/UploadModal/types';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ReactNode } from 'react';

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

describe('SimpleFilesTable', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockNewFiles: UploadFile[] = [
    { id: 'file-1', type: 'new', file: new File(['content1'], 'file1.jsonl') },
    { id: 'file-2', type: 'new', file: new File(['content2'], 'file2.jsonl') },
  ];

  const mockExistingFiles: UploadFile[] = [
    {
      id: 'existing-1',
      type: 'existing',
      file: {
        path: 'dataset/file1.jsonl',
        file_ref: 'ref1',
        size: 1024,
        file_url: 'https://example.com/file1.jsonl',
      },
    },
    {
      id: 'existing-2',
      type: 'existing',
      file: {
        path: 'dataset/file2.jsonl',
        file_ref: 'ref2',
        size: 2048,
        file_url: 'https://example.com/file2.jsonl',
      },
    },
  ];

  describe('rendering', () => {
    it('renders table with column headers', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getByText('Name')).toBeInTheDocument();
      expect(screen.getByText('Size')).toBeInTheDocument();
    });

    it('renders table with new files from context', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getByText('file1.jsonl')).toBeInTheDocument();
      expect(screen.getByText('file2.jsonl')).toBeInTheDocument();
    });

    it('renders table with existing files showing path', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockExistingFiles }),
      });

      expect(screen.getByText('dataset/file1.jsonl')).toBeInTheDocument();
      expect(screen.getByText('dataset/file2.jsonl')).toBeInTheDocument();
    });

    it('displays file sizes in formatted form', () => {
      // Use ``.json`` extensions so the default ``acceptableFileTypes``
      // filter (``['.json', '.jsonl']`` from the reducer's initial state)
      // doesn't hide these rows from the rendered table.
      const filesWithSize: UploadFile[] = [
        { id: 'file-1', type: 'new', file: new File(['x'.repeat(1024)], 'small.json') },
        {
          id: 'file-2',
          type: 'existing',
          file: {
            path: 'large.json',
            file_ref: 'ref1',
            size: 1048576,
            file_url: 'https://example.com/large.json',
          },
        },
      ];

      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: filesWithSize }),
      });

      expect(screen.getByText('1 kB')).toBeInTheDocument();
      expect(screen.getByText('1 MB')).toBeInTheDocument();
    });

    describe('multi-select (allowMultipleFileSelection=true)', () => {
      it('renders checkboxes for each file', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({ files: mockNewFiles, allowMultipleFileSelection: true }),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        expect(checkboxes).toHaveLength(2);
      });

      it('renders empty table when no files', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({ files: [], allowMultipleFileSelection: true }),
        });

        // Should still render headers
        expect(screen.getByText('Name')).toBeInTheDocument();
        expect(screen.getByText('Size')).toBeInTheDocument();
        // But no checkboxes
        expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
      });
    });
  });

  describe('file selection', () => {
    describe('multi-select (allowMultipleFileSelection=true)', () => {
      it('shows selected files as checked', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({
            files: mockNewFiles,
            selectedFiles: [mockNewFiles[0]],
            allowMultipleFileSelection: true,
          }),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        expect(checkboxes[0]).toBeChecked();
        expect(checkboxes[1]).not.toBeChecked();
      });

      it('shows all files as unchecked when none selected', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({
            files: mockNewFiles,
            selectedFiles: [],
            allowMultipleFileSelection: true,
          }),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        checkboxes.forEach((checkbox) => {
          expect(checkbox).not.toBeChecked();
        });
      });

      it('can show multiple files as selected', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({
            files: mockNewFiles,
            selectedFiles: [mockNewFiles[0], mockNewFiles[1]],
            allowMultipleFileSelection: true,
          }),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        expect(checkboxes[0]).toBeChecked();
        expect(checkboxes[1]).toBeChecked();
      });

      it('dispatches TOGGLE_FILE_SELECTION when checkbox is clicked', async () => {
        const user = userEvent.setup();
        const mockDispatch = vi.fn();
        render(<SimpleFilesTable />, {
          wrapper: createWrapper(
            { files: mockNewFiles, allowMultipleFileSelection: true },
            mockDispatch
          ),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        await user.click(checkboxes[0]);

        expect(mockDispatch).toHaveBeenCalledWith({
          type: 'TOGGLE_FILE_SELECTION',
          payload: mockNewFiles[0],
        });
      });

      it('dispatches correct file payload for existing files', async () => {
        const user = userEvent.setup();
        const mockDispatch = vi.fn();
        render(<SimpleFilesTable />, {
          wrapper: createWrapper(
            { files: mockExistingFiles, allowMultipleFileSelection: true },
            mockDispatch
          ),
        });

        const checkboxes = screen.getAllByRole('checkbox');
        await user.click(checkboxes[0]);

        expect(mockDispatch).toHaveBeenCalledWith({
          type: 'TOGGLE_FILE_SELECTION',
          payload: mockExistingFiles[0],
        });
      });

      it('exposes the file name as the accessible name of each checkbox', () => {
        render(<SimpleFilesTable />, {
          wrapper: createWrapper({
            files: mockNewFiles,
            allowMultipleFileSelection: true,
          }),
        });

        expect(screen.getByRole('checkbox', { name: 'file1.jsonl' })).toBeInTheDocument();
        expect(screen.getByRole('checkbox', { name: 'file2.jsonl' })).toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('displays error message when file error is present', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files: mockNewFiles,
          errors: { file: 'File is required' },
        }),
      });

      expect(screen.getByText('File is required')).toBeInTheDocument();
    });

    it('does not show error when errors object is empty', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files: mockNewFiles,
          errors: {},
        }),
      });

      expect(screen.queryByText(/is required/i)).not.toBeInTheDocument();
    });
  });

  describe('file type handling', () => {
    it('displays file name for new files', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getByText('file1.jsonl')).toBeInTheDocument();
    });

    it('displays file path for existing files', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockExistingFiles }),
      });

      expect(screen.getByText('dataset/file1.jsonl')).toBeInTheDocument();
    });

    it('handles mixed new and existing files', () => {
      const mixedFiles = [...mockNewFiles, ...mockExistingFiles];
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mixedFiles }),
      });

      // New files show name
      expect(screen.getByText('file1.jsonl')).toBeInTheDocument();
      // Existing files show path
      expect(screen.getByText('dataset/file1.jsonl')).toBeInTheDocument();
    });

    it('explains why unsupported files are disabled', () => {
      const files: UploadFile[] = [
        {
          id: 'config',
          type: 'existing',
          file: {
            path: 'eval/config.yaml',
            file_ref: 'ref-config',
            size: 100,
            file_url: 'https://example.com/config.yaml',
          },
        },
        {
          id: 'notes',
          type: 'existing',
          file: {
            path: 'eval/notes.txt',
            file_ref: 'ref-notes',
            size: 100,
            file_url: 'https://example.com/notes.txt',
          },
        },
      ];

      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files,
          acceptableFileTypes: ['.yml', '.yaml'],
          invalidFileMode: 'disable',
        }),
      });

      expect(screen.getByRole('radio', { name: 'eval/config.yaml' })).not.toBeDisabled();
      expect(screen.getByRole('radio', { name: 'eval/notes.txt' })).toBeDisabled();
      expect(
        screen.getByText(
          'Only .yml, .yaml files can be selected. Upload a supported file or choose a different fileset.'
        )
      ).toBeInTheDocument();
    });

    it('does not show disabled-file guidance when all visible files are selectable', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files: mockExistingFiles,
          acceptableFileTypes: ['.jsonl'],
          invalidFileMode: 'disable',
        }),
      });

      expect(
        screen.queryByText(/Upload a supported file or choose a different fileset/)
      ).not.toBeInTheDocument();
      expect(screen.getByRole('radio', { name: 'dataset/file1.jsonl' })).not.toBeDisabled();
      expect(screen.getByRole('radio', { name: 'dataset/file2.jsonl' })).not.toBeDisabled();
    });
  });

  describe('upload more files', () => {
    it('renders upload more files anchor', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getByText('Upload More Files')).toBeInTheDocument();
    });

    it('renders hidden file input with correct attributes', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files: mockNewFiles,
          acceptableFileTypes: ['.jsonl', '.json'],
        }),
      });

      const fileInput = screen.getByLabelText('Upload more files', {
        selector: 'input',
      }) as HTMLInputElement;
      expect(fileInput).toHaveClass('sr-only');
      expect(fileInput).toHaveAttribute('type', 'file');
      expect(fileInput).toHaveAttribute('accept', '.jsonl,.json');
    });

    it('dispatches SET_FILES when files are selected', async () => {
      const user = userEvent.setup();
      const mockDispatch = vi.fn();
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }, mockDispatch),
      });

      const fileInput = screen.getByLabelText('Upload more files', {
        selector: 'input',
      }) as HTMLInputElement;
      const newFile = new File(['new content'], 'newfile.jsonl');

      await user.upload(fileInput, newFile);

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'SET_FILES',
        payload: [
          {
            id: 'newfile.jsonl',
            type: 'new',
            file: newFile,
          },
        ],
      });
    });

    it('label is associated with file input', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      const fileInput = screen.getByLabelText('Upload more files', {
        selector: 'input',
      });

      expect(fileInput).toHaveAttribute('type', 'file');
    });

    it('file input does NOT have an id attribute (ref-based approach, no htmlFor)', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      const fileInput = screen.getByLabelText('Upload more files', {
        selector: 'input',
      });

      // The old implementation used id="upload-more-files"; the new approach uses a ref
      // and an aria-label, so no id should be present.
      expect(fileInput).not.toHaveAttribute('id');
    });

    it('file input has tabIndex={-1} to keep it out of the tab order', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      const fileInput = screen.getByLabelText('Upload more files', {
        selector: 'input',
      });

      expect(fileInput).toHaveAttribute('tabindex', '-1');
    });

    it('"Upload More Files" button is a native button (not a label element)', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      const button = screen.getByText('Upload More Files');
      // The button is now a KUI <Button kind="tertiary">, not a <label> element
      expect(button.closest('button')).not.toBeNull();
      expect(button.closest('label')).toBeNull();
    });
  });

  describe('single-select (allowMultipleFileSelection=false, default)', () => {
    it('renders a radio per file and no checkboxes', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getAllByRole('radio')).toHaveLength(2);
      expect(screen.queryAllByRole('checkbox')).toHaveLength(0);
    });

    it('shows the selected file as checked', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({
          files: mockNewFiles,
          selectedFiles: [mockNewFiles[0]],
        }),
      });

      const radios = screen.getAllByRole('radio');
      expect(radios[0]).toBeChecked();
      expect(radios[1]).not.toBeChecked();
    });

    it('dispatches TOGGLE_FILE_SELECTION when an unselected radio is clicked', async () => {
      const user = userEvent.setup();
      const mockDispatch = vi.fn();
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }, mockDispatch),
      });

      const radios = screen.getAllByRole('radio');
      await user.click(radios[1]);

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'TOGGLE_FILE_SELECTION',
        payload: mockNewFiles[1],
      });
    });

    it('does NOT dispatch when the already-selected radio is clicked', async () => {
      const user = userEvent.setup();
      const mockDispatch = vi.fn();
      render(<SimpleFilesTable />, {
        wrapper: createWrapper(
          { files: mockNewFiles, selectedFiles: [mockNewFiles[0]] },
          mockDispatch
        ),
      });

      const radios = screen.getAllByRole('radio');
      await user.click(radios[0]);

      expect(mockDispatch).not.toHaveBeenCalled();
    });

    it('dispatches correct payload for existing files', async () => {
      const user = userEvent.setup();
      const mockDispatch = vi.fn();
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockExistingFiles }, mockDispatch),
      });

      const radios = screen.getAllByRole('radio');
      await user.click(radios[0]);

      expect(mockDispatch).toHaveBeenCalledWith({
        type: 'TOGGLE_FILE_SELECTION',
        payload: mockExistingFiles[0],
      });
    });

    it('exposes the file name as the accessible name of each radio', () => {
      render(<SimpleFilesTable />, {
        wrapper: createWrapper({ files: mockNewFiles }),
      });

      expect(screen.getByRole('radio', { name: 'file1.jsonl' })).toBeInTheDocument();
      expect(screen.getByRole('radio', { name: 'file2.jsonl' })).toBeInTheDocument();
    });
  });
});
