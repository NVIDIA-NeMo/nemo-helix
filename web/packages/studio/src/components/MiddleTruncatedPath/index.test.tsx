// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MiddleTruncatedPath } from '@studio/components/MiddleTruncatedPath';
import { splitLastSegment } from '@studio/components/MiddleTruncatedPath/splitLastSegment';
import { render, screen } from '@studio/tests/util/render';

describe('splitLastSegment', () => {
  it.each([
    [
      'results/attempt-1/parquet-files/batch_00000.parquet',
      'results/attempt-1/parquet-files/',
      'batch_00000.parquet',
    ],
    ['rows.jsonl', '', 'rows.jsonl'],
    ['All 2 Parquet files in out/parquet-files/', 'All 2 Parquet files in out/', 'parquet-files/'],
    ['', '', ''],
  ])('splits %j', (path, head, tail) => {
    expect(splitLastSegment(path)).toEqual([head, tail]);
  });
});

describe('MiddleTruncatedPath', () => {
  it('names its container with the whole path and keeps the file name from shrinking', () => {
    const path = 'results/attempt-1/parquet-files/batch_00000.parquet';
    render(
      <div role="option" aria-selected={false}>
        <MiddleTruncatedPath>{path}</MiddleTruncatedPath>
      </div>
    );

    expect(screen.getByRole('option')).toHaveAccessibleName(path);
    expect(screen.getByTitle(path)).toBeInTheDocument();
    expect(screen.getByText('batch_00000.parquet')).toHaveClass('shrink-0');
  });
});
