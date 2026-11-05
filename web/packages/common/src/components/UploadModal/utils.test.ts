// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  isFilesetFileOutput,
  isBrowserFile,
  formatFileSize,
  getExistingFileId,
  sanitizeFilenameForDatasetName,
} from '@nemo/common/src/components/UploadModal/utils';
import { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';

describe('UploadModal utils', () => {
  describe('isFilesetFileOutput', () => {
    it('returns true for FilesetFileOutput objects', () => {
      const filesetFile: FilesetFileOutput = {
        path: 'test.jsonl',
        file_ref: 'ref123',
        size: 1024,
        file_url: 'https://example.com/test.jsonl',
      };

      expect(isFilesetFileOutput(filesetFile)).toBe(true);
    });

    it('returns false for File objects', () => {
      const file = new File(['content'], 'test.jsonl');
      expect(isFilesetFileOutput(file)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isFilesetFileOutput(undefined)).toBe(false);
    });
  });

  describe('isBrowserFile', () => {
    it('returns true for File objects', () => {
      const file = new File(['content'], 'test.jsonl');
      expect(isBrowserFile(file)).toBe(true);
    });

    it('returns false for FilesetFileOutput objects', () => {
      const filesetFile: FilesetFileOutput = {
        path: 'test.jsonl',
        file_ref: 'ref123',
        size: 1024,
        file_url: 'https://example.com/test.jsonl',
      };

      expect(isBrowserFile(filesetFile)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isBrowserFile(undefined)).toBe(false);
    });
  });

  describe('formatFileSize', () => {
    it('formats 0 bytes', () => {
      expect(formatFileSize(0)).toBe('0 Bytes');
    });

    it('formats bytes', () => {
      expect(formatFileSize(100)).toBe('100 Bytes');
      expect(formatFileSize(500)).toBe('500 Bytes');
    });

    it('formats kilobytes', () => {
      expect(formatFileSize(1024)).toBe('1 kB');
      expect(formatFileSize(1024 * 5)).toBe('5 kB');
      expect(formatFileSize(1024 * 5.5)).toBe('5.5 kB');
    });

    it('formats megabytes', () => {
      expect(formatFileSize(1024 * 1024)).toBe('1 MB');
      expect(formatFileSize(1024 * 1024 * 2.5)).toBe('2.5 MB');
    });

    it('formats gigabytes', () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe('1 GB');
      expect(formatFileSize(1024 * 1024 * 1024 * 1.5)).toBe('1.5 GB');
    });

    it('formats terabytes', () => {
      expect(formatFileSize(1024 * 1024 * 1024 * 1024)).toBe('1 TB');
      expect(formatFileSize(1024 * 1024 * 1024 * 1024 * 2.25)).toBe('2.25 TB');
    });

    it('rounds to 2 decimal places', () => {
      expect(formatFileSize(1536)).toBe('1.5 kB');
      expect(formatFileSize(1024 * 1.567)).toBe('1.57 kB');
    });
  });

  describe('getExistingFileId', () => {
    it('returns a unique id combining file_ref and path', () => {
      const filesetFile: FilesetFileOutput = {
        path: 'test.jsonl',
        file_ref: 'ref123',
        size: 1024,
        file_url: 'https://example.com/test.jsonl',
      };

      expect(getExistingFileId(filesetFile)).toBe('ref123-test.jsonl');
    });

    it('handles files with different paths but same file_ref', () => {
      const file1: FilesetFileOutput = {
        path: 'original.jsonl',
        file_ref: 'same-ref',
        size: 1024,
        file_url: 'https://example.com/original.jsonl',
      };

      const file2: FilesetFileOutput = {
        path: 'renamed.jsonl',
        file_ref: 'same-ref',
        size: 1024,
        file_url: 'https://example.com/renamed.jsonl',
      };

      const id1 = getExistingFileId(file1);
      const id2 = getExistingFileId(file2);

      expect(id1).toBe('same-ref-original.jsonl');
      expect(id2).toBe('same-ref-renamed.jsonl');
      expect(id1).not.toBe(id2);
    });

    it('handles files with nested paths', () => {
      const filesetFile: FilesetFileOutput = {
        path: 'folder/subfolder/test.jsonl',
        file_ref: 'ref456',
        size: 2048,
        file_url: 'https://example.com/folder/subfolder/test.jsonl',
      };

      expect(getExistingFileId(filesetFile)).toBe('ref456-folder/subfolder/test.jsonl');
    });

    it('handles files with special characters in path', () => {
      const filesetFile: FilesetFileOutput = {
        path: 'test-file_name (1).jsonl',
        file_ref: 'ref789',
        size: 512,
        file_url: 'https://example.com/test-file_name (1).jsonl',
      };

      expect(getExistingFileId(filesetFile)).toBe('ref789-test-file_name (1).jsonl');
    });
  });

  describe('sanitizeFilenameForDatasetName', () => {
    it('removes file extension', () => {
      expect(sanitizeFilenameForDatasetName('my-dataset.json')).toBe('my-dataset');
      expect(sanitizeFilenameForDatasetName('data.jsonl')).toBe('data');
      expect(sanitizeFilenameForDatasetName('file.txt')).toBe('file');
    });

    it('lowercases and keeps valid characters (alphanumeric, dots, underscores, dashes)', () => {
      expect(sanitizeFilenameForDatasetName('my-dataset-123.json')).toBe('my-dataset-123');
      expect(sanitizeFilenameForDatasetName('my_dataset_v2.json')).toBe('my_dataset_v2');
      expect(sanitizeFilenameForDatasetName('dataset.v1.0.json')).toBe('dataset.v1.0');
      expect(sanitizeFilenameForDatasetName('Dataset123.json')).toBe('dataset123');
      expect(sanitizeFilenameForDatasetName('MyDataset.JSON')).toBe('mydataset');
    });

    it('replaces invalid characters with underscores', () => {
      expect(sanitizeFilenameForDatasetName('my dataset.json')).toBe('my_dataset');
      expect(sanitizeFilenameForDatasetName('dataset@2024.json')).toBe('dataset_2024');
      expect(sanitizeFilenameForDatasetName('file(1).json')).toBe('file_1_');
      expect(sanitizeFilenameForDatasetName('data#set!.json')).toBe('data_set_');
    });

    it('handles multiple consecutive invalid characters', () => {
      expect(sanitizeFilenameForDatasetName('my   dataset.json')).toBe('my___dataset');
      expect(sanitizeFilenameForDatasetName('data!!!set.json')).toBe('data___set');
    });

    it('handles filenames with multiple dots', () => {
      expect(sanitizeFilenameForDatasetName('data.backup.json')).toBe('data.backup');
      expect(sanitizeFilenameForDatasetName('file.v1.0.json')).toBe('file.v1.0');
    });

    it('handles files without extensions', () => {
      expect(sanitizeFilenameForDatasetName('my-dataset')).toBe('my-dataset');
      expect(sanitizeFilenameForDatasetName('data_file')).toBe('data_file');
    });

    it('returns empty string for empty or invalid-only filenames', () => {
      expect(sanitizeFilenameForDatasetName('.json')).toBe('');
      expect(sanitizeFilenameForDatasetName('   .json')).toBe('');
      expect(sanitizeFilenameForDatasetName('!!!')).toBe('');
    });

    it('handles special characters that need escaping', () => {
      expect(sanitizeFilenameForDatasetName('data$set.json')).toBe('data_set');
      expect(sanitizeFilenameForDatasetName('file^name.json')).toBe('file_name');
      expect(sanitizeFilenameForDatasetName('test&data.json')).toBe('test_data');
    });

    it('handles unicode and international characters', () => {
      expect(sanitizeFilenameForDatasetName('données.json')).toBe('donn_es');
      // Filenames with only unicode characters result in only underscores, so we return empty string
      expect(sanitizeFilenameForDatasetName('数据集.json')).toBe('');
      expect(sanitizeFilenameForDatasetName('файл.json')).toBe('');
    });

    it('preserves leading and trailing valid characters', () => {
      expect(sanitizeFilenameForDatasetName('_dataset.json')).toBe('_dataset');
      expect(sanitizeFilenameForDatasetName('-data-.json')).toBe('-data-');
      expect(sanitizeFilenameForDatasetName('123data.json')).toBe('123data');
    });
  });
});
