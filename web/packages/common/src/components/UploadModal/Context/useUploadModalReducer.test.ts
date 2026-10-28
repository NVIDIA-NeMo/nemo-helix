// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  uploadModalReducer,
  initialState,
  UploadModalState,
} from '@nemo/common/src/components/UploadModal/Context/useUploadModalReducer';
import { UploadFile } from '@nemo/common/src/components/UploadModal/types';

describe('uploadModalReducer', () => {
  describe('SET_FILES with auto-select', () => {
    it.each(['disable', 'hide'] as const)(
      'should NOT auto-select a single unsupported file when invalidFileMode is %s',
      (invalidFileMode) => {
        const readme: UploadFile = {
          id: 'readme',
          type: 'existing',
          file: { path: 'README.md', size: 1, file_ref: 'r' } as never,
        };
        const state: UploadModalState = {
          ...initialState,
          acceptableFileTypes: ['.json', '.jsonl', '.csv'],
          invalidFileMode,
        };

        expect(
          uploadModalReducer(state, { type: 'SET_FILES', payload: [readme] }).selectedFiles
        ).toEqual([]);

        const csv: UploadFile = {
          ...readme,
          id: 'csv',
          file: { ...readme.file, path: 'DATA.CSV' } as never,
        };
        expect(
          uploadModalReducer(state, { type: 'SET_FILES', payload: [csv] }).selectedFiles
        ).toEqual([csv]);
      }
    );

    it('should auto-select a single file when allowMultipleFileSelection is false', () => {
      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(initialState, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.files).toHaveLength(1);
      expect(newState.selectedFiles).toHaveLength(1);
      expect(newState.selectedFiles[0]).toBe(mockFile);
    });

    it('should NOT auto-select when multiple files are added', () => {
      const mockFile1: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content1'], 'test1.jsonl', { type: 'application/jsonl' }),
      };
      const mockFile2: UploadFile = {
        id: 'file2',
        type: 'new',
        file: new File(['content2'], 'test2.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(initialState, {
        type: 'SET_FILES',
        payload: [mockFile1, mockFile2],
      });

      expect(newState.files).toHaveLength(2);
      expect(newState.selectedFiles).toHaveLength(0);
    });

    it('should NOT auto-select when allowMultipleFileSelection is true', () => {
      const stateWithMultiSelect: UploadModalState = {
        ...initialState,
        allowMultipleFileSelection: true,
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(stateWithMultiSelect, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.files).toHaveLength(1);
      expect(newState.selectedFiles).toHaveLength(0);
    });

    it('should preserve existing selection when adding more files', () => {
      const mockFile1: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content1'], 'test1.jsonl', { type: 'application/jsonl' }),
      };
      const mockFile2: UploadFile = {
        id: 'file2',
        type: 'new',
        file: new File(['content2'], 'test2.jsonl', { type: 'application/jsonl' }),
      };

      // First, add one file (should auto-select)
      const stateWithOneFile = uploadModalReducer(initialState, {
        type: 'SET_FILES',
        payload: [mockFile1],
      });

      expect(stateWithOneFile.selectedFiles).toHaveLength(1);

      // Then add another file
      const finalState = uploadModalReducer(stateWithOneFile, {
        type: 'SET_FILES',
        payload: [mockFile2],
      });

      expect(finalState.files).toHaveLength(2);
      expect(finalState.selectedFiles).toHaveLength(1);
      expect(finalState.selectedFiles[0]).toBe(mockFile1);
    });
  });

  describe('RESET action', () => {
    it('should preserve configuration props when resetting', () => {
      const customState: UploadModalState = {
        ...initialState,
        acceptableFileTypes: ['.jsonl', '.csv', '.parquet'],
        acceptableFileSize: 100 * 1024 * 1024,
        allowMultipleFileSelection: true,
        files: [
          {
            id: 'file1',
            type: 'new',
            file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
          },
        ],
        selectedFiles: [
          {
            id: 'file1',
            type: 'new',
            file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
          },
        ],
      };

      const resetState = uploadModalReducer(customState, { type: 'RESET' });

      // Configuration props should be preserved
      expect(resetState.acceptableFileTypes).toEqual(['.jsonl', '.csv', '.parquet']);
      expect(resetState.acceptableFileSize).toBe(100 * 1024 * 1024);
      expect(resetState.allowMultipleFileSelection).toBe(true);

      // But files and selections should be cleared
      expect(resetState.files).toHaveLength(0);
      expect(resetState.selectedFiles).toHaveLength(0);
      expect(resetState.dataset).toBeUndefined();
    });
  });

  describe('SET_DATASET action', () => {
    it('should clear files and selectedFiles when setting a dataset', () => {
      const stateWithFiles: UploadModalState = {
        ...initialState,
        files: [
          {
            id: 'file1',
            type: 'new',
            file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
          },
        ],
        selectedFiles: [
          {
            id: 'file1',
            type: 'new',
            file: new File(['content'], 'test.jsonl', { type: 'application/jsonl' }),
          },
        ],
      };

      const newState = uploadModalReducer(stateWithFiles, {
        type: 'SET_DATASET',
        payload: {
          type: 'new',
          name: 'test-dataset',
        },
      });

      expect(newState.files).toHaveLength(0);
      expect(newState.selectedFiles).toHaveLength(0);
      expect(newState.dataset).toEqual({ type: 'new', name: 'test-dataset' });
    });
  });

  describe('SET_FILES with auto-naming dataset', () => {
    it('should auto-set dataset name from first file when name is empty', () => {
      const stateWithNewDataset: UploadModalState = {
        ...initialState,
        dataset: {
          type: 'new',
          name: '',
        },
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'my-dataset-file.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(stateWithNewDataset, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.dataset).toEqual({
        type: 'new',
        name: 'my-dataset-file',
      });
      expect(newState.files).toHaveLength(1);
    });

    it('should NOT auto-set dataset name when name is already set', () => {
      const stateWithNamedDataset: UploadModalState = {
        ...initialState,
        dataset: {
          type: 'new',
          name: 'existing-name',
        },
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'different-file.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(stateWithNamedDataset, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.dataset).toEqual({
        type: 'new',
        name: 'existing-name',
      });
    });

    it('should NOT auto-set dataset name when dataset type is existing', () => {
      const stateWithExistingDataset: UploadModalState = {
        ...initialState,
        dataset: {
          type: 'existing',
          dataset: {
            id: 'dataset-id',
            name: 'existing-dataset',
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
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'test-file.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(stateWithExistingDataset, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.dataset).toEqual({
        type: 'existing',
        dataset: {
          id: 'dataset-id',
          name: 'existing-dataset',
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
      });
    });

    it('should NOT auto-set dataset name when files already exist', () => {
      const stateWithFilesAndDataset: UploadModalState = {
        ...initialState,
        dataset: {
          type: 'new',
          name: '',
        },
        files: [
          {
            id: 'file0',
            type: 'new',
            file: new File(['content'], 'existing-file.jsonl', { type: 'application/jsonl' }),
          },
        ],
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'new-file.jsonl', { type: 'application/jsonl' }),
      };

      const newState = uploadModalReducer(stateWithFilesAndDataset, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.dataset).toEqual({
        type: 'new',
        name: '',
      });
      expect(newState.files).toHaveLength(2);
    });

    it('should handle files without extensions correctly', () => {
      const stateWithNewDataset: UploadModalState = {
        ...initialState,
        dataset: {
          type: 'new',
          name: '',
        },
      };

      const mockFile: UploadFile = {
        id: 'file1',
        type: 'new',
        file: new File(['content'], 'my-dataset-file', { type: 'application/octet-stream' }),
      };

      const newState = uploadModalReducer(stateWithNewDataset, {
        type: 'SET_FILES',
        payload: [mockFile],
      });

      expect(newState.dataset).toEqual({
        type: 'new',
        name: 'my-dataset-file',
      });
    });
  });
});
