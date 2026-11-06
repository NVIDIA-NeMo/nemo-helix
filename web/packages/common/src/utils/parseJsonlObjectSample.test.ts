// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  buildRowsAndKeysFromJsonlSample,
  formatJsonlSampleCellValue,
  JSONL_SAMPLE_RAW_LINE_KEY,
  JSONL_SAMPLE_SCALAR_KEY,
  labelForJsonlSampleColumnKey,
} from '@nemo/common/src/utils/parseJsonlObjectSample';

describe('buildRowsAndKeysFromJsonlSample', () => {
  it('parses JSONL objects and unions keys in first-seen order', () => {
    const text = ['{"a":1,"b":2}', '{"b":3,"c":4}'].join('\n');
    const { rows, columnKeys } = buildRowsAndKeysFromJsonlSample(text);
    expect(columnKeys).toEqual(['a', 'b', 'c']);
    expect(rows).toHaveLength(2);
    expect(rows[0].values).toEqual({ a: 1, b: 2 });
    expect(rows[1].values).toEqual({ b: 3, c: 4 });
  });

  it('wraps non-object JSON in a synthetic column', () => {
    const { rows, columnKeys } = buildRowsAndKeysFromJsonlSample('[1,2]');
    expect(columnKeys).toEqual([JSONL_SAMPLE_SCALAR_KEY]);
    expect(rows[0].values[JSONL_SAMPLE_SCALAR_KEY]).toBe('[1,2]');
  });

  it('puts invalid JSON lines in a synthetic raw column', () => {
    const { rows, columnKeys } = buildRowsAndKeysFromJsonlSample('not json');
    expect(columnKeys).toEqual([JSONL_SAMPLE_RAW_LINE_KEY]);
    expect(rows[0].values[JSONL_SAMPLE_RAW_LINE_KEY]).toBe('not json');
  });

  it('preserves invalid JSON line text after filtering blank lines', () => {
    const { rows } = buildRowsAndKeysFromJsonlSample('  not json  \n\n{"ok":true}');
    expect(rows[0].values[JSONL_SAMPLE_RAW_LINE_KEY]).toBe('  not json  ');
  });
});

describe('formatJsonlSampleCellValue', () => {
  it('handles undefined', () => {
    expect(formatJsonlSampleCellValue(undefined)).toBe('');
  });

  it('handles null', () => {
    expect(formatJsonlSampleCellValue(null)).toBe('null');
  });

  it('returns strings unchanged', () => {
    expect(formatJsonlSampleCellValue('hello')).toBe('hello');
  });

  it('converts numbers and booleans', () => {
    expect(formatJsonlSampleCellValue(42)).toBe('42');
    expect(formatJsonlSampleCellValue(true)).toBe('true');
  });

  it('stringifies nested objects', () => {
    expect(formatJsonlSampleCellValue({ x: 1 })).toBe('{"x":1}');
  });

  it('falls back to String(value)', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(formatJsonlSampleCellValue(circular)).toBe('[object Object]');
  });
});

describe('labelForJsonlSampleColumnKey', () => {
  it('maps synthetic keys to labels', () => {
    expect(labelForJsonlSampleColumnKey(JSONL_SAMPLE_SCALAR_KEY)).toBe('(non-object JSON)');
    expect(labelForJsonlSampleColumnKey(JSONL_SAMPLE_RAW_LINE_KEY)).toBe('(unparsed line)');
    expect(labelForJsonlSampleColumnKey('prompt')).toBe('prompt');
  });
});
