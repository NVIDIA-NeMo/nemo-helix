// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseDataFile, serializeDataFile } from '@studio/components/FileRowEditor/parse';
import { ROW_ID_KEY } from '@studio/components/FileRowEditor/types';

describe('parseDataFile (csv)', () => {
  it('splits rows on mixed line endings but keeps quoted line breaks', () => {
    const rows = parseDataFile('name\r\nx\ny\r"a\r\nb"\n', 'csv');
    expect(rows.map((row) => row.name)).toEqual(['x', 'y', 'a\r\nb']);
  });

  it('keeps zero-padded identifiers as strings instead of coercing to numbers', () => {
    const rows = parseDataFile('id,qty\n007,3\n00501,0\n42,-5', 'csv');

    expect(rows[0].id).toBe('007');
    expect(rows[1].id).toBe('00501');
    // Genuine integers still coerce.
    expect(rows[0].qty).toBe(3);
    expect(rows[1].qty).toBe(0);
    expect(rows[2].id).toBe(42);
    expect(rows[2].qty).toBe(-5);
  });

  it('drops only trailing blank lines, preserving single-column empty values', () => {
    // A single "value" column with a legitimately empty middle row and a trailing newline.
    const rows = parseDataFile('value\na\n\nb\n', 'csv');

    expect(rows.map((row) => row.value)).toEqual(['a', '', 'b']);
  });
});

describe('serializeDataFile', () => {
  const rows = [
    { [ROW_ID_KEY]: 1, name: 'Acme, Inc.', count: 3, meta: { vip: true } },
    { [ROW_ID_KEY]: 2, name: 'Globex', count: 0, meta: null },
  ];

  it('strips the synthetic row id and escapes CSV cells', () => {
    const csv = serializeDataFile(rows, 'csv');

    expect(csv).toBe(
      ['name,count,meta', '"Acme, Inc.",3,"{""vip"":true}"', 'Globex,0,'].join('\n')
    );
    expect(csv).not.toContain(ROW_ID_KEY);
  });

  it('emits one compact object per line for jsonl without the row id', () => {
    const jsonl = serializeDataFile(rows, 'jsonl');

    expect(jsonl).toBe(
      [
        '{"name":"Acme, Inc.","count":3,"meta":{"vip":true}}',
        '{"name":"Globex","count":0,"meta":null}',
      ].join('\n')
    );
  });

  it('round-trips csv content through parse and serialize', () => {
    const original = 'name,count,meta\n"Acme, Inc.",3,"{""vip"":true}"\nGlobex,0,';
    const reparsed = parseDataFile(serializeDataFile(parseDataFile(original, 'csv'), 'csv'), 'csv');

    expect(reparsed).toEqual([
      { name: 'Acme, Inc.', count: 3, meta: '{"vip":true}' },
      { name: 'Globex', count: 0, meta: '' },
    ]);
  });
});
