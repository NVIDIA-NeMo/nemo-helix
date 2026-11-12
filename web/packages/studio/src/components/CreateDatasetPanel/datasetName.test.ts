// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getDatasetNameError, inferDatasetName } from '@studio/components/CreateDatasetPanel/datasetName';

const err = (name: string, taken = false) => getDatasetNameError(name, { taken, workspace: 'default' });

describe('getDatasetNameError (spec 3.1, 3.2)', () => {
  it('returns undefined for valid names, including dots and trailing/repeated dots the predicate permits', () => {
    for (const n of ['ab', 'alpaca', 'my-data.v2', 'a1', 'data.', 'a..b', 'a'.repeat(63)]) {
      expect(err(n), n).toBeUndefined();
    }
  });

  it('maps each invalid case to the spec 3.2 message (spec 14 name row)', () => {
    expect(err('')).toBe('Enter a name for this Dataset');
    expect(err('1data')).toBe('Names must start with a lowercase letter');
    expect(err('Data')).toBe('Names must start with a lowercase letter');
    expect(err('-abc')).toBe('Names must start with a lowercase letter');
    expect(err('a')).toBe('Names must be at least 2 characters');
    expect(err('a'.repeat(64))).toBe('Names must be 63 characters or fewer');
    expect(err('data--train')).toBe("Names can't contain two hyphens in a row");
    expect(err('data-')).toBe("Names can't end with a hyphen");
    expect(err('my_data')).toBe('Use lowercase letters, digits, hyphens, and dots only');
    expect(err('my data')).toBe('Use lowercase letters, digits, hyphens, and dots only');
    expect(err('dataSet')).toBe('Use lowercase letters, digits, hyphens, and dots only');
  });

  it('applies the first matching rule in table order when several rules match', () => {
    expect(err('Data--x-')).toBe('Names must start with a lowercase letter'); // start beats `--`, trailing hyphen
    expect(err('ab--c-')).toBe("Names can't contain two hyphens in a row"); // `--` beats trailing hyphen
    expect(err('a' + 'b'.repeat(62) + '--')).toBe('Names must be 63 characters or fewer'); // length beats `--`
    expect(err('ab_-')).toBe("Names can't end with a hyphen"); // trailing hyphen beats invalid chars
  });

  it('duplicate message wins over format errors and names workspace', () => {
    expect(getDatasetNameError('taken-name', { taken: true, workspace: 'team-a' })).toBe(
      'A Dataset named taken-name already exists in team-a',
    );
    expect(getDatasetNameError('Bad Name', { taken: true, workspace: 'team-a' })).toBe(
      'A Dataset named Bad Name already exists in team-a',
    );
  });
});

describe('inferDatasetName (spec 3.3)', () => {
  it('lowercases, replaces runs of other characters with one hyphen, collapses hyphens, keeps dots', () => {
    expect(inferDatasetName('Stanford_Alpaca')).toBe('stanford-alpaca');
    expect(inferDatasetName('My  Data!!set')).toBe('my-data-set');
    expect(inferDatasetName('a_-_b')).toBe('a-b');
    expect(inferDatasetName('a--b')).toBe('a-b');
    expect(inferDatasetName('v1.2-Final')).toBe('v1.2-final');
    expect(inferDatasetName('nemotron-personas-dataset-en_us')).toBe('nemotron-personas-dataset-en-us');
  });

  it('strips leading and trailing hyphens and dots', () => {
    expect(inferDatasetName('.-a-.')).toBe('a');
    expect(inferDatasetName('--abc..')).toBe('abc');
    expect(inferDatasetName('___')).toBe('');
    expect(inferDatasetName('')).toBe('');
  });

  it('does not guarantee validity: digit-start and single-char seeds stay invalid', () => {
    expect(inferDatasetName('1data')).toBe('1data');
    expect(getDatasetNameError(inferDatasetName('1data'), { taken: false, workspace: 'w' })).toBe(
      'Names must start with a lowercase letter',
    );
    expect(inferDatasetName('A')).toBe('a');
    expect(getDatasetNameError(inferDatasetName('A'), { taken: false, workspace: 'w' })).toBe(
      'Names must be at least 2 characters',
    );
  });

  it('truncates to 63 then strips trailing hyphens/dots exposed by truncation', () => {
    expect(inferDatasetName('a'.repeat(100))).toBe('a'.repeat(63));
    expect(inferDatasetName('a'.repeat(62) + '-b')).toBe('a'.repeat(62));
    expect(inferDatasetName('a'.repeat(62) + '.b')).toBe('a'.repeat(62));
    expect(inferDatasetName('a'.repeat(61) + '-.b')).toBe('a'.repeat(61));
  });
});
