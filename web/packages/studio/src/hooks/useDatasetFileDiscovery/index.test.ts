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
    ['train_part-0.jsonl', 'training'],
    ['train.json', 'training'],
    ['validation.jsonl', 'validation'],
    ['val.jsonl', 'validation'],
    ['val_part-0.json', 'validation'],
    ['dev.jsonl', 'validation'],
    ['train/part-0.jsonl', 'training'],
    ['training/rows.json', 'training'],
    ['dev/rows.jsonl', 'validation'],
    ['/validation.jsonl', 'validation'],
  ])('reads %s as %s data from its name or directory', (path, expected) => {
    // A file named for each split keeps the fallback out of the way. The anchors share the
    // case's extension so a .json case is not dropped beside .jsonl files.
    const ext = path.endsWith('.json') ? '.json' : '.jsonl';
    const [training, validation] = split(path, `validation_anchor${ext}`, `train_anchor${ext}`);
    expect((expected === 'training' ? training : validation).includes(path)).toBe(true);
  });

  // A pattern matches from the start of the name, case-sensitively.
  it.each([
    'my_train_rows.jsonl',
    'pretrain.jsonl',
    'constraints.jsonl',
    'eval.jsonl',
    'retrieval.jsonl',
    'Train.jsonl',
    'Training/rows.jsonl',
    'rows.jsonl',
    'test.jsonl',
    'eval_beir/corpus.jsonl',
    'additional/train.json',
    'train/nested/part-0.jsonl',
    'data/train/part-0.jsonl',
  ])('leaves %s out when other files are named for a split', (path) => {
    expect(split(path, 'train.jsonl').flat()).not.toContain(path);
  });

  it('picks both splits and leaves other files out', () => {
    expect(
      split(
        'training.jsonl',
        'validation.jsonl',
        'eval.jsonl',
        'test.jsonl',
        'README.md',
        'eval_beir/corpus.jsonl'
      )
    ).toEqual([['training.jsonl'], ['validation.jsonl']]);
  });

  it('treats train and training as the same split', () => {
    expect(split('training.jsonl', 'train.jsonl')).toEqual([['train.jsonl', 'training.jsonl'], []]);
  });

  it('without names, takes every root .jsonl file as training data', () => {
    expect(split('b.jsonl', 'a.jsonl', 'c.json', 'nested/d.jsonl')).toEqual([
      ['a.jsonl', 'b.jsonl'],
      [],
    ]);
  });

  it('without names, does not take .json or nested files as training data', () => {
    expect(split('rows.json', 'data/c.jsonl', 'data/a.jsonl')).toEqual([[], []]);
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
