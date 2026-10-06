// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { estimateRows } from '@studio/hooks/useDatasetFormat';

const rows = (count: number) => Array.from({ length: count }, () => '{"a":1}\n').join('');

describe('estimateRows', () => {
  it('counts exactly when the whole file was read', () => {
    const content = rows(10);
    expect(estimateRows(content, content.length)).toEqual({ rows: 10, estimated: false });
  });

  it('scales a size-capped preview up by the file size', () => {
    const preview = rows(10);
    expect(estimateRows(preview, preview.length * 50)).toEqual({ rows: 500, estimated: true });
  });

  it('ignores blank lines', () => {
    expect(estimateRows('{"a":1}\n\n{"a":2}\n', 16).rows).toBe(2);
  });
});
