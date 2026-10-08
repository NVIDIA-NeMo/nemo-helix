// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FilesetFileOutput } from '@nemo/sdk/generated/platform/schema';
import { partitionDatasetFiles } from '@studio/hooks/useDatasetFileDiscovery';

const file = (path: string): FilesetFileOutput => ({
  file_ref: path,
  file_url: `https://example.test/${path}`,
  path,
  size: 1024,
});

const paths = (files: FilesetFileOutput[]) => files.map((f) => f.path);

describe('partitionDatasetFiles', () => {
  it('returns empty buckets for an empty file list', () => {
    expect(partitionDatasetFiles([])).toEqual({
      training: [],
      validation: [],
      unmatchedRootJsonl: [],
      unmatchedRootJson: [],
    });
  });

  it('matches files inside training/ and validation/ subfolders', () => {
    const result = partitionDatasetFiles([
      file('training/data.jsonl'),
      file('validation/data.jsonl'),
    ]);
    expect(paths(result.training)).toEqual(['training/data.jsonl']);
    expect(paths(result.validation)).toEqual(['validation/data.jsonl']);
    expect(result.unmatchedRootJsonl).toEqual([]);
  });

  it('matches the alternate train/, val/, dev/ subfolders', () => {
    const result = partitionDatasetFiles([
      file('train/a.jsonl'),
      file('val/b.jsonl'),
      file('dev/c.jsonl'),
    ]);
    expect(paths(result.training)).toEqual(['train/a.jsonl']);
    // val/ and dev/ both map to validation in customizer's heuristic.
    expect(paths(result.validation).sort()).toEqual(['dev/c.jsonl', 'val/b.jsonl']);
  });

  it('matches root-level files by train*/training* / val*/validation*/dev* patterns', () => {
    const result = partitionDatasetFiles([
      file('train_a.jsonl'),
      file('training-data.jsonl'),
      file('val_b.jsonl'),
      file('validation_b.jsonl'),
      file('dev_c.jsonl'),
    ]);
    expect(paths(result.training).sort()).toEqual(['train_a.jsonl', 'training-data.jsonl']);
    expect(paths(result.validation).sort()).toEqual([
      'dev_c.jsonl',
      'val_b.jsonl',
      'validation_b.jsonl',
    ]);
    expect(result.unmatchedRootJsonl).toEqual([]);
  });

  it('parks unmatched root .jsonl files in unmatchedRootJsonl (caller decides fallback)', () => {
    const result = partitionDatasetFiles([file('mystery.jsonl')]);
    expect(result.training).toEqual([]);
    expect(result.validation).toEqual([]);
    expect(paths(result.unmatchedRootJsonl)).toEqual(['mystery.jsonl']);
  });

  it('parks every unmatched root .jsonl in unmatchedRootJsonl (the hook decides whether to claim them)', () => {
    // The pure partition function never claims root-fallback files itself —
    // useDatasetFileDiscovery does that based on whether train/val matched
    // anything. This test pins the partition contract; the hook-level
    // auto-claim of multiple files is covered in the hook integration tests.
    const result = partitionDatasetFiles([file('a.jsonl'), file('b.jsonl')]);
    expect(result.training).toEqual([]);
    expect(result.validation).toEqual([]);
    expect(paths(result.unmatchedRootJsonl).sort()).toEqual(['a.jsonl', 'b.jsonl']);
  });

  it('parks unmatched root .json (non-jsonl) files in unmatchedRootJson, not unmatchedRootJsonl', () => {
    // .json files without a train/val pattern are not eligible for the
    // Customizer lone-root fallback (.jsonl-only), but are captured separately
    // so native Unsloth can claim them when apply_chat_template is disabled.
    const result = partitionDatasetFiles([file('thing.json')]);
    expect(result.unmatchedRootJsonl).toEqual([]);
    expect(paths(result.unmatchedRootJson)).toEqual(['thing.json']);
  });

  it('ignores files with non-dataset extensions', () => {
    const result = partitionDatasetFiles([
      file('README.md'),
      file('training/.gitkeep'),
      file('training/data.png'),
    ]);
    expect(result.training).toEqual([]);
    expect(result.validation).toEqual([]);
    expect(result.unmatchedRootJsonl).toEqual([]);
  });

  it('claims .json files inside subfolders, not just .jsonl', () => {
    const result = partitionDatasetFiles([file('training/data.json')]);
    expect(paths(result.training)).toEqual(['training/data.json']);
  });

  it('claims nested files (more than one path segment past the subfolder)', () => {
    const result = partitionDatasetFiles([file('training/sub/data.jsonl')]);
    expect(paths(result.training)).toEqual(['training/sub/data.jsonl']);
  });

  it('does not match arbitrary subfolder names', () => {
    const result = partitionDatasetFiles([file('not-train/data.jsonl')]);
    expect(result.training).toEqual([]);
    expect(result.validation).toEqual([]);
    expect(result.unmatchedRootJsonl).toEqual([]);
  });

  it('coexists: subfolder hits + root patterns + unmatched root', () => {
    const result = partitionDatasetFiles([
      file('training/data.jsonl'),
      file('val_b.jsonl'),
      file('mystery.jsonl'),
      file('README.md'),
    ]);
    expect(paths(result.training)).toEqual(['training/data.jsonl']);
    expect(paths(result.validation)).toEqual(['val_b.jsonl']);
    expect(paths(result.unmatchedRootJsonl)).toEqual(['mystery.jsonl']);
  });
});
