// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ContentType } from '@nemo/common/src/components/CodeEditor/constants';
import {
  fromFilesetUrl,
  inferJsonContentType,
  isJsonFile,
  parseFilesetUrl,
} from '@nemo/common/src/components/DatasetFileSelect/utils';

describe('FileInput utils', () => {
  describe('parseFilesetUrl', () => {
    it('parses fileset URL to components', () => {
      const result = parseFilesetUrl('fileset://my-workspace/my-dataset/train.csv');
      expect(result).toEqual({
        workspace: 'my-workspace',
        name: 'my-dataset',
        path: 'train.csv',
      });
    });

    it('handles nested file paths', () => {
      const result = parseFilesetUrl('fileset://workspace/dataset/folder/subfolder/file.jsonl');
      expect(result).toEqual({
        workspace: 'workspace',
        name: 'dataset',
        path: 'folder/subfolder/file.jsonl',
      });
    });

    it('parses fileset:// URL with hash path', () => {
      const result = parseFilesetUrl('fileset://my-ws/my-ds#outputs/part.csv');
      expect(result).toEqual({
        workspace: 'my-ws',
        name: 'my-ds',
        path: 'outputs/part.csv',
      });
    });

    it('parses short-form workspace/fileset#path ref', () => {
      const result = parseFilesetUrl('default/qa-dataset#training/training.jsonl');
      expect(result).toEqual({
        workspace: 'default',
        name: 'qa-dataset',
        path: 'training/training.jsonl',
      });
    });

    it('returns null for invalid URLs - missing filepath', () => {
      const result = parseFilesetUrl('fileset://workspace/dataset');
      expect(result).toBeNull();
    });

    it('returns null for invalid URLs - wrong prefix', () => {
      const result = parseFilesetUrl('hf://datasets/org/dataset/file.csv');
      expect(result).toBeNull();
    });

    it('returns null for malformed URLs', () => {
      const result = parseFilesetUrl('invalid-url');
      expect(result).toBeNull();
    });
  });

  describe('fromFilesetUrl', () => {
    it('parses fileset URL to FileListItem', () => {
      const result = fromFilesetUrl('fileset://my-workspace/my-dataset/train.csv');
      expect(result).toEqual({
        path: 'train.csv',
        url: 'fileset://my-workspace/my-dataset/train.csv',
      });
    });

    it('handles nested file paths', () => {
      const result = fromFilesetUrl('fileset://workspace/dataset/folder/subfolder/file.jsonl');
      expect(result).toEqual({
        path: 'folder/subfolder/file.jsonl',
        url: 'fileset://workspace/dataset/folder/subfolder/file.jsonl',
      });
    });

    it('returns null for invalid URLs', () => {
      expect(fromFilesetUrl('invalid-url')).toBeNull();
      expect(fromFilesetUrl('hf://datasets/org/dataset/file.csv')).toBeNull();
    });
  });

  describe('inferJsonContentType', () => {
    it('returns JSONL for .jsonl files', () => {
      expect(inferJsonContentType('data.jsonl')).toBe(ContentType.JSONL);
      expect(inferJsonContentType('path/to/file.jsonl')).toBe(ContentType.JSONL);
    });

    it('returns JSON for .json files', () => {
      expect(inferJsonContentType('data.json')).toBe(ContentType.JSON);
      expect(inferJsonContentType('path/to/config.json')).toBe(ContentType.JSON);
    });

    it('is case-insensitive', () => {
      expect(inferJsonContentType('DATA.JSONL')).toBe(ContentType.JSONL);
      expect(inferJsonContentType('Config.JSON')).toBe(ContentType.JSON);
    });

    it('returns null for non-JSON files', () => {
      expect(inferJsonContentType('data.csv')).toBeNull();
      expect(inferJsonContentType('document.txt')).toBeNull();
      expect(inferJsonContentType('image.png')).toBeNull();
    });

    it('returns null for files without extension', () => {
      expect(inferJsonContentType('README')).toBeNull();
      expect(inferJsonContentType('Makefile')).toBeNull();
    });

    it('handles multiple dots in filename', () => {
      expect(inferJsonContentType('my.data.file.json')).toBe(ContentType.JSON);
      expect(inferJsonContentType('backup.2024.jsonl')).toBe(ContentType.JSONL);
    });
  });

  describe('isJsonFile', () => {
    it('returns true for JSON content type', () => {
      expect(isJsonFile(ContentType.JSON)).toBe(true);
    });

    it('returns true for JSONL content type', () => {
      expect(isJsonFile(ContentType.JSONL)).toBe(true);
    });

    it('returns false for null', () => {
      expect(isJsonFile(null)).toBe(false);
    });

    it('returns false for other content types', () => {
      expect(isJsonFile('csv')).toBe(false);
      expect(isJsonFile('text')).toBe(false);
      expect(isJsonFile('')).toBe(false);
    });
  });
});
