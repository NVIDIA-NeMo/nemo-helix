// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { uploadNameSeed } from '@studio/components/CreateDatasetPanel/uploadNameSeed';

describe('uploadNameSeed', () => {
  it('derives the seed from the folder, else the first loose file', () => {
    // no files
    expect(uploadNameSeed([])).toBe('');
    // single loose file: only the final extension is removed
    expect(uploadNameSeed(['data.final.jsonl'])).toBe('data.final');
    // first loose file wins, extension-less files are kept whole
    expect(uploadNameSeed(['train.jsonl', 'val.jsonl'])).toBe('train');
    expect(uploadNameSeed(['README'])).toBe('README');
    // a common top-level folder, with nested paths preserved beneath it
    expect(uploadNameSeed(['my-data/a.jsonl', 'my-data/sub/deep/b.jsonl'])).toBe('my-data');
    // folder takes priority over loose files that precede it
    expect(uploadNameSeed(['loose.txt', 'my-data/a.jsonl', 'my-data/b.jsonl'])).toBe('my-data');
    // different top-level folders have no common folder: fall back to the first loose file
    expect(uploadNameSeed(['one/a.jsonl', 'two/b.jsonl', 'loose.csv'])).toBe('loose');
    // folders only, but no common one: nothing to name from
    expect(uploadNameSeed(['one/a.jsonl', 'two/b.jsonl'])).toBe('');
  });
});
