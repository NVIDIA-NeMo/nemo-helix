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

/** The training and validation paths, as Customizer's select_split_files returns them. */
const split = (...paths: string[]) => {
  const { training, validation } = partitionDatasetFiles(paths.map(file));
  return [training.map((f) => f.path), validation.map((f) => f.path)];
};

// The same cases as packages/nhx_customization_common/tests/test_dataset_files.py,
// so Studio shows the files training reads.
describe('partitionDatasetFiles', () => {
  it.each([
    ['train.jsonl', 'training'],
    ['training.jsonl', 'training'],
    ['my_train_rows.jsonl', 'training'],
    ['Pretrain-Corpus.jsonl', 'training'],
    ['trainval.jsonl', 'training'],
    ['validation.jsonl', 'validation'],
    ['val.jsonl', 'validation'],
    ['eval.jsonl', 'validation'],
    ['dev.jsonl', 'validation'],
    ['train/part-0.jsonl', 'training'],
    ['Training/rows.jsonl', 'training'],
    ['dev/rows.jsonl', 'validation'],
  ])('reads %s as %s data from its name or directory', (path, expected) => {
    // A file named for neither split keeps the fallback out of the way.
    const [training, validation] = split(path, 'validation_anchor.jsonl', 'train_anchor.jsonl');
    expect((expected === 'training' ? training : validation).includes(path)).toBe(true);
  });

  it.each([
    'test.jsonl',
    'eval_beir/corpus.jsonl',
    'additional/train.json',
    'data/train/part-0.jsonl',
  ])('leaves %s out when other files are named for a split', (path) => {
    expect(split(path, 'train.jsonl').flat()).not.toContain(path);
  });

  it('picks both splits and leaves other files out', () => {
    expect(
      split('training.jsonl', 'eval.jsonl', 'test.jsonl', 'README.md', 'eval_beir/corpus.jsonl')
    ).toEqual([['training.jsonl'], ['eval.jsonl']]);
  });

  it('treats train and training as the same split', () => {
    expect(split('training.jsonl', 'train.jsonl')).toEqual([['train.jsonl', 'training.jsonl'], []]);
  });

  it('without names, takes every root data file as training data', () => {
    expect(split('b.jsonl', 'a.jsonl', 'nested/c.jsonl')).toEqual([['a.jsonl', 'b.jsonl'], []]);
  });

  it('without root data files, takes the nested ones as training data', () => {
    expect(split('data/c.jsonl', 'data/a.jsonl')).toEqual([['data/a.jsonl', 'data/c.jsonl'], []]);
  });

  it('with only a validation file, has no training data', () => {
    expect(split('validation.jsonl', 'rows.jsonl')).toEqual([[], ['validation.jsonl']]);
  });

  it('drops .json beside .jsonl and uses it alone', () => {
    expect(split('train.json', 'training.jsonl')).toEqual([['training.jsonl'], []]);
    expect(split('train.json')).toEqual([['train.json'], []]);
  });

  it('ignores files that are not JSON or JSONL', () => {
    expect(split('README.md', 'training/.gitkeep', 'training/data.png')).toEqual([[], []]);
  });
});
